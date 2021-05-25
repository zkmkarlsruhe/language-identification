import os
import shutil
import argparse
import time
from datetime import datetime
from yaml import load

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import src.models as models
from src.utils.training_utils import *
from src.audio.generators import AugBatchGenerator
from src.audio.augment import AudioAugmenter

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

	# when using on-the-fly augmentation create the augmentation object 
	augmenter = None
	if augment:
		augmenter = AudioAugmenter(audio_length_s, fs)

	# create generator objects
	train_gen_obj = AugBatchGenerator(source=train_dir, target_length_s=audio_length_s,
									languages=languages, batch_size=batch_size, 
									augmenter=augmenter)
	val_gen_obj = AugBatchGenerator(source=val_dir, target_length_s=audio_length_s,
									languages=languages, batch_size=batch_size)

	# progress bar information
	show_progress = True
	train_num_batches = train_gen_obj.count_batches()
	val_num_batches = val_gen_obj.count_batches()

	# Training loop
	best_val_acc = 0.0
	for epoch in range(1, num_epochs+1):
                
		print('===== EPOCH ', str(epoch), ' ======')

		# train
		train_results = run_epoch(model, train_gen_obj, training=True, optimizer=optimizer,
									show_progress=show_progress, num_batches=train_num_batches)
		train_acc, train_loss, train_recall, train_precision = train_results

		# validate
		val_results = run_epoch(model, val_gen_obj, training=False, 
								show_progress=show_progress, num_batches=val_num_batches)
		val_acc, val_loss, val_recall, val_precision = val_results

		# log data
		lr = round(float(get_value(optimizer.learning_rate)), 6)
		logs = {'epoch': epoch, 'learning_rate': lr, 
				'train_acc': train_acc, 'train_loss': train_loss, 
				'train_rec': train_recall, 'train_pre': train_precision, 
				'val_acc': val_acc, 'val_loss': val_loss,
				'val_rec': val_recall, 'val_pre': val_precision}
		write_csv(os.path.join(log_dir, 'log.csv'), epoch, logs)

		# save model
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			model_name = os.path.join(log_dir, 'model' + "_" + str(epoch))
			model_predict = get_saved_model_function(model, dims=(1, audio_length_s*fs,1))
			model.save(model_name, signatures={'serving_default': model_predict})
		


if __name__ == "__main__": 
	tf.config.list_physical_devices('GPU')
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', default=None)
	parser.add_argument('--config', default="config.yaml")
	cli_args = parser.parse_args()

	log_dir = os.path.join("logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
	print("Logging to {}".format(log_dir))

	# copy models & config for later
	shutil.copytree("src/models", os.path.join(log_dir, "models"))
	shutil.copy(cli_args.config, log_dir)

	model_file_name = train(cli_args.config, log_dir, cli_args.model_path)
	print("Best model at: ", model_file_name)

