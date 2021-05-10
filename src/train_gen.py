import os
import shutil
import argparse
import time
from datetime import datetime
from yaml import load

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Progbar

import models
from utils.training_utils import *
from audio.generators import LIDGenerator
from audio.augment import AudioAugmenter
from audio.utils import pad_with_silence
from audio.features import normalize

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def batch_gen(generator, batch_size, augmenter=None, fs=None, desired_audio_length_s=10):
	x_batch = []
	y_batch = []
	i=0
	while True:
		try:
			x,y = next(generator)
			if augmenter:
				x = augmenter.augment_audio_array(x, fs)
				x = pad_with_silence(x, desired_audio_length_s *fs)
			x = normalize(x)
			x_batch.append(x)
			y_batch.append(y)
			i += 1
			if i == batch_size:
				x_arr = np.asarray(x_batch)
				y_arr = np.asarray(y_batch)
				yield x_arr, y_arr
				i = 0
				x_batch = []
				y_batch = []
		except StopIteration as e:
				if len(x_batch) > 0:
					yield x_batch, y_batch
				break
					
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
	fs = config["sample_rate"]
	audio_length_s = config["audio_length_s"]

	model_class = getattr(models, config["model"])
	model = model_class.create_model(config)
	optimizer = Adam(lr=config["learning_rate"])
	model.compile()
	print(model.summary())
	augmenter = AudioAugmenter(fs)

	def run_epoch(batch_generator, num_batches, training):
		''' train or validate over an epoch
		'''
		loss_fn = CategoricalCrossentropy()
		accuracy_fn = CategoricalAccuracy()
		accuracy_epoch = []
		loss_epoch = []

		if training:
			print('______ TRAINING ______')
		else:
			print('_____ VALIDATION ______')

		metrics_names = ['loss', 'acc']
		pb = Progbar(num_batches, stateful_metrics=metrics_names)

		# Iterate over the batches of a dataset
		for x_batch_train, y_batch_train in batch_generator:
			x_batch_train = np.expand_dims(x_batch_train, axis=-1)

			with tf.GradientTape() as tape:
				# run the model
				logits = model(x_batch_train, training=training)

				# calculate loss & metrics
				loss_value = loss_fn(y_batch_train, logits)
				accuracy_fn.update_state(y_batch_train, logits)

			# optimize
			if training:
				grads = tape.gradient(loss_value, model.trainable_weights)
				optimizer.apply_gradients(zip(grads, model.trainable_weights))
				
			# report loss & metrics per batch
			accuracy = accuracy_fn.result()
			values=[('loss', loss_value), ('acc', accuracy)]
			pb.add(1, values=values)
			loss_epoch.append(loss_value)
			accuracy_epoch.append(accuracy)

		# report loss & metrics per epoch
		epoch_loss = np.mean(loss_epoch)
		epoch_acc = np.mean(accuracy_epoch)
		print ('Epoch ended with: \tmean(loss): %.4f \tmean(acc): %.4f\n' % (epoch_loss, epoch_acc))

		return epoch_acc, epoch_loss


	# Epochs Loop
	for epoch in range(num_epochs):
                

		# create Generators
		train_gen_obj = LIDGenerator(source=train_dir, target_length_s=10, shuffle=True,
								languages=languages)
		#train_generator = batch_gen(train_gen_obj.get_generator(), batch_size, augmenter, fs, audio_length_s)
		train_generator = batch_gen(train_gen_obj.get_generator(), batch_size)

		val_gen_obj = LIDGenerator(source=val_dir, target_length_s=10, shuffle=True,
								languages=languages)
		val_generator = batch_gen(val_gen_obj.get_generator(), batch_size)
		print('===== EPOCH ', str(epoch), ' ======')

		train_num_batches = train_gen_obj.get_num_files() // batch_size
		val_num_batches = val_gen_obj.get_num_files() // batch_size

		train_acc, train_loss = run_epoch(train_generator, train_num_batches, training=True)
		val_acc, val_loss = run_epoch(val_generator, val_num_batches, training=False)
		logs = {'train_acc': train_acc, 'train_loss': train_loss, 
		 'val_acc': val_acc, 'val_loss': val_loss}

		model.save(os.path.join(log_dir, 'model' + "_" + str(epoch + 1)))
		write_csv(os.path.join(log_dir, 'log.csv'), optimizer, epoch, logs)
		

	# Do evaluation on model with best validation accuracy
	best_epoch = np.argmax(history.history["val_categorical_accuracy"])
	print("Log files: ", log_dir)
	print("Best epoch: ", best_epoch)
	return


if __name__ == "__main__": 
	tf.config.list_physical_devices('GPU')
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', default=None)
	parser.add_argument('--config', default="config.yaml")
	cli_args = parser.parse_args()

	log_dir = os.path.join("logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
	print("Logging to {}".format(log_dir))

	# copy models & config for later
	shutil.copytree("models", os.path.join(log_dir, "models"))
	shutil.copy(cli_args.config, log_dir)

	model_file_name = train(cli_args.config, log_dir, cli_args.model_path)
	print("Best model at: ", model_file_name)

