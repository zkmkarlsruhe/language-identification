"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under Simplified BSD License.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.backend import get_value
from tensorflow.keras.losses import CategoricalCrossentropy as CrossLoss
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy, CategoricalCrossentropy
from tensorflow.keras.utils import Progbar


from kapre.composed import get_stft_magnitude_layer
from kapre.composed import get_melspectrogram_layer
from kapre.composed import get_log_frequency_spectrogram_layer


def create_dataset_from_set_of_files(ds_dir, languages):
	
	# assure languages are sorted alphanumerically
	languages = sorted(languages)

	# create a file path dataset from all directories specified
	glob_list = [os.path.join(ds_dir, lang,'*.wav') for lang in languages]
	list_ds = tf.data.Dataset.list_files(glob_list)

	# create a dataset yielding audio and categorical label
	def process_path(file_path):
		# read the wav file
		x = tf.io.read_file(file_path)
		x, _ = tf.audio.decode_wav(x)
		# x = tf.squeeze(x, axis=-1)
		
		# get label and convert to categorical 
		label = tf.strings.split(file_path, os.sep)[-2]
		y = tf.cast(tf.equal(label, languages), tf.float32)
		return x, y
	labeled_ds = list_ds.map(process_path)

	return labeled_ds


def tf_normalize(signal):
	"""
	normalize a float signal to have a maximum absolute value of 1.0
	"""
	highest = tf.math.abs(tf.math.reduce_max(signal))
	lowest = tf.math.abs(tf.math.reduce_min(signal))
	abs_max = tf.math.maximum(highest, lowest)
	return tf.math.divide(signal, abs_max)


def get_feature_layer(feature_type, feature_nu, sample_rate):
	if feature_type == 'stft':
		m = get_stft_magnitude_layer(n_fft=feature_nu*2, name='stft_deb')
	elif feature_type == 'mel':
		m = get_melspectrogram_layer(n_mels=feature_nu, sample_rate=sample_rate, 
									name='mel_deb')
	elif feature_type == 'fbank':
		m = get_log_frequency_spectrogram_layer(log_n_bins=feature_nu, sample_rate=sample_rate, 
									name='fbank_deb')
	else:
		print("create_model: Unknown feature type!")
		return None
	return m


def write_csv(logging_dir, epoch, logs={}):
		with open(logging_dir, mode='a') as log_file:
			log_file_writer = csv.writer(log_file, delimiter=',')
			if epoch == 0:
				row = list(logs.keys())
				log_file_writer.writerow(row)
			row_vals = [round(x, 6) for x in list(logs.values())]
			log_file_writer.writerow(row_vals)


class CustomCSVCallback(Callback):
	def __init__(self, logging_dir):
		self._logging_dir = logging_dir
	def on_epoch_end(self, epoch, logs={}):
		logs['learning_rate'] = float(get_value(self.model.optimizer.learning_rate))
		write_csv(self._logging_dir, epoch, logs)


def get_saved_model_function(model, dims=(None, None, 1)):
	@tf.function(input_signature=[tf.TensorSpec(dims, dtype=tf.float32)])
	def model_predict(input_1):
		return {'outputs': model(input_1, training=False)}
	return model_predict


def visualize_results(history, config, log_dir):
	epochs = config["num_epochs"]
	acc = history.history['categorical_accuracy']
	val_acc = history.history['val_categorical_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs_range = range(epochs)
	
	plt.figure(figsize=(8, 8))
	plt.subplot(2, 1, 1)
	plt.plot(epochs_range, acc, label='Training Accuracy')
	plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')

	plt.subplot(2, 1, 2)
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')

	plt.savefig(os.path.join(log_dir, "training.png"))


##### Old Stuff ######

def run_epoch(model, batch_generator_obj, training=False, optimizer=None, show_progress=False, num_batches=32):
	"""Train or validate a model with a given generator.

	Args:
		model (Keras.Model): keras model
		batch_generator (generator): generator object that yields batches of samples
		training (bool, optional): Whether to train the model or not. Defaults to False.
		optimizer (Keras.Optimizer, optional): optimizer that is applied when training is True. Defaults to None.
		show_progress (bool, optional): whether to show the progress bar. Defaults to False.
		num_batches (int): maximum number of batches (only used for progress bar)

	Returns:
		[list]: a list of metrics
	"""
	
	if training:
		print('______ TRAINING ______')
	else:
		print('_____ VALIDATION ______')

	# get the generator function
	batch_generator = batch_generator_obj.get_generator()

	# metrics
	loss_fn = CrossLoss()
	metric_accuracy = CategoricalAccuracy()
	metric_recall = Recall()
	metric_precision = Precision()
	metric_loss = CategoricalCrossentropy()
	metrics_names = ['loss', 'acc', 'rec', 'pre']
	pb = Progbar(num_batches, stateful_metrics=metrics_names)

	# iterate over the batches of a dataset
	for x_batch, y_batch in batch_generator:

		# audio samples are expected to have one channel
		x_batch = np.expand_dims(x_batch, axis=-1)

		with tf.GradientTape() as tape:
			logits = model(x_batch, training=training)
			loss_value = loss_fn(y_batch, logits)
			
		# update metrics
		metric_accuracy.update_state(y_batch, logits)
		metric_precision.update_state(y_batch, logits)
		metric_recall.update_state(y_batch, logits)
		metric_loss.update_state(y_batch, logits)

		# optimize
		if training:
			grads = tape.gradient(loss_value, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))
			
		# report loss & metrics
		accuracy 	= metric_accuracy.result().numpy()
		recall 		= metric_recall.result().numpy()
		precision 	= metric_precision.result().numpy()
		loss 		= metric_loss.result().numpy()

		if show_progress:
			values = [ ('loss', loss), ('acc', accuracy), 
						('rec', recall), ('pre', precision) ]
			pb.add(1, values=values)

	return accuracy, loss, recall, precision
