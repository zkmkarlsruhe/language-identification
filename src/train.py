import os
import shutil
import argparse
import time
from datetime import datetime
from yaml import load

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Progbar

import models
from utils.training_utils import *
from audio.generators import AugBatchGenerator
from audio.augment import AudioAugmenter

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

					
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
	augment = config["augment"]

	# create or load the model
	if model_path:
		model = load_model(model_path)
	else:
		model_class = getattr(models, config["model"])
		model = model_class.create_model(config)
		optimizer = Adam(lr=config["learning_rate"])

	model.compile()
	print(model.summary())
	
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
		for x_batch, y_batch in batch_generator:
			x_batch = np.expand_dims(x_batch, axis=-1)

			with tf.GradientTape() as tape:
				# run the model
				logits = model(x_batch, training=training)

				# calculate loss & metrics
				loss_value = loss_fn(y_batch, logits)
				accuracy_fn.update_state(y_batch, logits)

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
		

	# create generator objects
	if augment:
		augmenter = AudioAugmenter(fs)
		train_gen_obj = AugBatchGenerator(source=train_dir, target_length_s=audio_length_s,
										languages=languages, batch_size=batch_size, 
										augmenter=augmenter, fs=fs)
	else:
		train_gen_obj = AugBatchGenerator(source=train_dir, target_length_s=audio_length_s,
										languages=languages, batch_size=batch_size)
	val_gen_obj = AugBatchGenerator(source=val_dir, target_length_s=audio_length_s,
									languages=languages)

	# get generators
	val_generator = val_gen_obj.get_generator()
	train_generator = train_gen_obj.get_generator()

	# dataset info
	train_num_batches = train_gen_obj.get_num_files() // batch_size
	val_num_batches = val_gen_obj.get_num_files() // batch_size

	# Training loop
	best_val_acc = 0.0
	for epoch in range(1, num_epochs+1):
                
		print('===== EPOCH ', str(epoch), ' ======')

		# train
		train_acc, train_loss = run_epoch(train_generator, train_num_batches, training=True)

		# validate
		val_acc, val_loss = run_epoch(val_generator, val_num_batches, training=False)

		# log data
		lr = round(float(get_value(optimizer.learning_rate)), 6)
		logs = {'epoch': epoch, 'learning_rate': lr, 
				'train_acc': train_acc, 'train_loss': train_loss, 
				'val_acc': val_acc, 'val_loss': val_loss}
		write_csv(os.path.join(log_dir, 'log.csv'), logs)

		# save model
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			model.save(os.path.join(log_dir, 'model' + "_" + str(epoch)))

		train_gen_obj.reset()
		val_gen_obj.reset()
		
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

