"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under Simplified BSD License.
"""

import os
import shutil
import argparse
from datetime import datetime
from yaml import load

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.models import load_model

import src.models as models
from src.utils.training_utils import CustomCSVCallback, get_saved_model_function, visualize_results
from src.utils.training_utils import create_dataset_from_set_of_files, tf_normalize
from src.audio.augment import AudioAugmenter



def train(config_path, log_dir):

	# Config
	config = load(open(config_path, "rb"))
	if config is None:
		print("Please provide a config.")
	train_dir = config["train_dir"]
	val_dir = config["val_dir"]
	batch_size = config["batch_size"]
	languages = config["languages"]
	num_epochs = config["num_epochs"]
	sample_rate = config["sample_rate"]
	audio_length_s = config["audio_length_s"]
	augment = config["augment"]
	learning_rate = config["learning_rate"]
	model_name = config["model"]
	model_path = config["model_path"]

	# create or load the model
	if model_path != "":
		model = load_model(model_path)
	else:
		model_class = getattr(models, model_name)
		model = model_class.create_model(config)
		optimizer = Adam(lr=learning_rate)
		model.compile(optimizer=optimizer,
						loss=CategoricalCrossentropy(),
						metrics=[Recall(), Precision(), CategoricalAccuracy()])
	print(model.summary())

	# load the dataset
	train_ds = create_dataset_from_set_of_files(
						ds_dir=train_dir, languages=languages)
	val_ds = create_dataset_from_set_of_files(
						ds_dir=val_dir, languages=languages)

	# Optional augmentation of the training set
	## Note: tf.py_function allows to construct a graph but code is executed in python (may be slow)
	if augment:
		augmenter = AudioAugmenter(audio_length_s, sample_rate)
		# process a single audio array (note: dataset needs to be batched later on)
		def process_aug(audio, label):
			augmented_audio = augmenter.augment_audio(audio.numpy())
			tensor_audio = tf.convert_to_tensor(augmented_audio, dtype=tf.float32)
			return tensor_audio, label
		aug_wav = lambda x,y: tf.py_function(process_aug, [x, y], [tf.float32, tf.float32])
		train_ds = train_ds.map(aug_wav, num_parallel_calls=tf.data.experimental.AUTOTUNE)

	# normalize audio and expand by one dimension (as required by feature extraction)
	def process(audio, label):
		audio = tf_normalize(audio)
		audio = tf.expand_dims(audio, axis=-1)
		return audio, label
	train_ds = train_ds.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	val_ds = val_ds.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE)

	# batch and prefetch data
	train_ds = train_ds.batch(batch_size)
	val_ds = val_ds.batch(batch_size)
	train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
	val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

	# Training Callbacks
	tensorboard_callback = TensorBoard(log_dir=log_dir, write_images=True)
	csv_logger_callback = CustomCSVCallback(os.path.join(log_dir, "log.csv"))
	reduce_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3,
										 verbose=1, min_lr=0.000001, min_delta=0.001)
	checkpoint_filename = os.path.join(log_dir, "trained_models", "model.{epoch:02d}")
	model_checkpoint_callback = ModelCheckpoint(checkpoint_filename, save_best_only=True, verbose=1,
												monitor="val_categorical_accuracy",
												save_weights_only=False)
	early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode="min")
	
	# comment callbacks that you don't care about
	callbacks = [
		# tensorboard_callback, 
		csv_logger_callback, 
		reduce_on_plateau, 
		# model_checkpoint_callback,
		# early_stopping_callback, 
		]

	# Training
	history = model.fit(x=train_ds, epochs=num_epochs,
						callbacks=callbacks, validation_data=val_ds)


	# TODO Do evaluation on model with best validation accuracy
	visualize_results(history, config, log_dir)
	best_epoch = np.argmax(history.history["val_categorical_accuracy"])
	print("Log files: ", log_dir)
	print("Best epoch: ", best_epoch)
	checkpoint_filename.replace("{epoch:02d}", "{:02d}".format(best_epoch))
	print("Best model at: ", checkpoint_filename)

	return model, best_epoch


if __name__ == "__main__": 

	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default="config_train.yaml", 
						help="Path to the required config file.")
	cli_args = parser.parse_args()
	
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:	
		try:
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
		except RuntimeError as e:
			print(e)

	# copy models & config for later
	log_dir = os.path.join("logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
	print("Logging to {}".format(log_dir))
	shutil.copytree("src/models", os.path.join(log_dir, "models"))
	shutil.copy(cli_args.config, log_dir)

	# train and save the best model as SavedModel
	model, best_epoch = train(cli_args.config, log_dir)
	saved_model_path = os.path.join(log_dir, "model_" + str(best_epoch))
	model.save(saved_model_path, signatures=get_saved_model_function(model))

	#TODO visualize the training process and save as png
	#TODO convert model to saved model

