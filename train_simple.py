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
from src.utils.training_utils import CustomCSVCallback, get_saved_model_function,
from src.utils.training_utils import create_dataset_from_set_of_files, tf_normalize
from src.audio.augment import AudioAugmenter



def train(config_path, log_dir, model_path):

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

	# create or load the model
	if model_path:
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
	train_ds = train_ds.batch(batch_size)
	val_ds = val_ds.batch(batch_size)

	# Optional augmentation of the training set
	if augment:
		augmenter = AudioAugmenter(audio_length_s, sample_rate)
		def process_aug(audio, label):
			audio = augmenter.augment_audio_array(audio)
			return audio, label
		# train_ds.map(process_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE)

	# normalize audio
	def process(audio, label):
		audio = tf_normalize(audio)
		return audio, label
	train_ds = train_ds.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	val_ds = val_ds.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE)

	# prefetch data
	train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
	val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

	# Training Callbacks
	tensorboard_callback = TensorBoard(log_dir=log_dir, write_images=True)
	csv_logger_callback = CustomCSVCallback(os.path.join(log_dir, "log.csv"))
	reduce_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1,
										 verbose=1, min_lr=0.000001, min_delta=0.001)
	checkpoint_filename = os.path.join(log_dir, "trained_models", "weights.{epoch:02d}")
	model_checkpoint_callback = ModelCheckpoint(checkpoint_filename, save_best_only=True, verbose=1,
												monitor="val_categorical_accuracy",
												save_weights_only=False)
	#early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode="min")
	
	callbacks = [tensorboard_callback, csv_logger_callback, reduce_on_plateau, model_checkpoint_callback,
								# early_stopping_callback, 
								]

	# Training
	history = model.fit(x=train_ds, epochs=num_epochs,
						callbacks=callbacks, validation_data=val_ds)

	# visualize_results(history, config)

	# Do evaluation on model with best validation accuracy
	best_epoch = np.argmax(history.history["val_categorical_accuracy"])
	print("Log files: ", log_dir)
	print("Best epoch: ", best_epoch)
	return checkpoint_filename.replace("{epoch:02d}", "{:02d}".format(best_epoch))


if __name__ == "__main__": 

	tf.config.list_physical_devices('GPU')

	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default="config_train.yaml")
	parser.add_argument('--model_path', default=None, help="Path to a trained model for retraining")
	cli_args = parser.parse_args()

	log_dir = os.path.join("logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
	print("Logging to {}".format(log_dir))

	# copy models & config for later
	shutil.copytree("src/models", os.path.join(log_dir, "models"))
	shutil.copy(cli_args.config, log_dir)

	model_file_name = train(cli_args.config, log_dir, cli_args.model_path)
	print("Best model at: ", model_file_name)

