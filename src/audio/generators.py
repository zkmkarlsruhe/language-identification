"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under Simplified BSD License.
"""

import os
import random
import numpy as np
import fnmatch
import scipy.io.wavfile as wav
import tensorflow as tf

from .augment import AudioAugmenter
from .utils import pad_with_silence
from .features import normalize

def recursive_glob(path, pattern):
	for root, dirs, files in os.walk(path):
		for basename in files:
			if fnmatch.fnmatch(basename, pattern):
				filename = os.path.abspath(os.path.join(root, basename))
				if os.path.isfile(filename):
					yield filename


class AudioGenerator(object):
	def __init__(self, source, target_length_s, shuffle=True, run_only_once=True,
				dtype="float32", minimum_length=0.5):
		"""
		A class for generating audio samples of equal length either from directory or file.

		Args:
			source (str): may be a file or directory containing audio files
			target_length_s (float):  the length of the desired audio chunks in seconds
			shuffle (bool, optional): whether to shuffle the list. Defaults to True.
			run_only_once (bool, optional): whether to stop after all chunks have been yielded once. Defaults to True.
			dtype (str, optional): output data tpe. Defaults to "float32".
			minimum_length (float, optional):  minimum length of the audio chunk in percentage. Defaults to 0.5.
		"""

		self.source = source
		self.shuffle = shuffle
		self.target_length_s = target_length_s
		self.run_only_once = run_only_once
		self.dtype = dtype
		self.minimum_length = minimum_length
		if os.path.isdir(self.source):
			files = []
			files.extend(recursive_glob(self.source, "*.wav"))
			if shuffle:
				np.random.shuffle(files)
		else:
			files = [self.source]
		self.files = files

	def get_generator(self):
		"""
		returns a generator that iterates over the source directory or file
		the generator yields audio chunks of desired target length, the sampling frequency and file name
		"""
		file_counter = 0
		while True:
			file = self.files[file_counter]
			try:
				# read a file and calculate according parameters
				fs, audio = wav.read(file)
				file_name = file.split('/')[-1]
				target_length = self.target_length_s * fs
				num_segments = int(len(audio) // target_length)

				# for all segments create slices of target length
				for i in range(0, num_segments):
					slice_start = int(i * target_length)
					slice_end = int(slice_start + target_length)
					rest = len(audio) - slice_start
					# if we have only one segment left and there is at least
					#  minimum_length data pad it with silence
					if i == num_segments:
						if rest >= target_length * self.minimum_length:
							chunk = pad_with_silence(audio[slice_start:], target_length)
						else:
							break
					else:
						chunk = audio[slice_start:slice_end]
					chunk = chunk.astype(dtype=self.dtype)
					yield [chunk, fs, file_name]

			except Exception as e:
				print("AudioGenerator Exception: ", e, file)
				pass

			finally:
				file_counter += 1
				if file_counter >= len(self.files):
					if self.run_only_once:
						break
					if os.path.isdir(self.source) and self.shuffle:
						np.random.shuffle(self.files)
					file_counter = 0

	def get_num_files(self):
		return len(self.files)


class LIDGenerator(object):
	def __init__(self, source, target_length_s, shuffle=True, languages=[],
				dtype="float32"):
		"""	
		A class for generating labeled audio samples of equal length from directories.
		Labels for each language are generated in alphanumerical order (chinese, english, kabyle, ...).

		Args:
			source (str): directory containing directories for each language
			target_length_s (float): the length of the desired audio chunks in seconds
			shuffle (bool, optional): whether to shuffle the list of paths before yielding. Defaults to True.
			languages (list, optional): a list of the sub directories containing audio. Defaults to [].
			dtype (str, optional): output data type. Defaults to "float32".
		"""
		self.source = source
		self.shuffle = shuffle
		self.target_length_s = target_length_s
		self.languages = sorted(languages)
		self.num_classes = len(languages)
		self.dtype = dtype
		if len(languages) == 0:
			print("Please provide at least one language")
		self.reset()

	def reset(self):
		self.active_generators = [i for i in range(0, self.num_classes)]
		self.generators = [AudioGenerator(source=os.path.join(self.source, language), 
									target_length_s=self.target_length_s, dtype=self.dtype,
									shuffle=self.shuffle, run_only_once=True) 
									for language in self.languages]
		self.pipelines = [gen.get_generator() for gen in self.generators]

	def get_generator(self):
		"""
		returns a generator that exhausts all internal language specific generators
		the generator yields audio and label data
		"""
		while True:
			if len(self.active_generators) == 0:
				break
			rand_gen_index = random.choice(self.active_generators)
			try:
				audio, fs, name = next(self.pipelines[rand_gen_index])
				label = tf.keras.utils.to_categorical(rand_gen_index, self.num_classes)
				yield (audio, label)
			except StopIteration as e:
				self.active_generators.remove(rand_gen_index)

	def get_num_files(self):
		return sum([gen.get_num_files() for gen in self.generators])


class AugBatchGenerator():

	def __init__(self, source, target_length_s, languages=[], batch_size=32, augmenter=None):
		"""
		A wrapper class around LidGenerator for generating batches.
		Batches can be augmented if necessary.

		Args:
			source (str): directory containing directories for each language
			target_length_s (float): the length of the desired audio chunks in seconds
			languages (list, optional): a list of the sub directories containing audio. Defaults to [].
			batch_size (int, optional): amount of samples per batch. Defaults to 32.
			augmenter (AudioAugmenter, optional): augment object. Defaults to None.
		"""

		self.source = source
		self.target_length_s = target_length_s
		self.languages = sorted(languages)
		self.num_classes = len(languages)
		self.batch_size = batch_size
		self.augmenter = augmenter
		self.reset()

	def reset(self):
		self.generatorObj = LIDGenerator(source=self.source, languages=self.languages,
										target_length_s=self.target_length_s, shuffle=True)
		self.generator = self.generatorObj.get_generator()

	def get_generator(self):
		x_batch = []
		y_batch = []
		i=0

		def process_batch(x_batch, y_batch):
			if self.augmenter:
				x_batch = self.augmenter.augment_audio_array(x_batch)
			x_batch = [normalize(item) for item in x_batch]
			return np.asarray(x_batch), np.asarray(y_batch)

		while True:

			try:
				x,y = next(self.generator)
				x_batch.append(x)
				y_batch.append(y)
				i += 1
				if i == self.batch_size:
					yield process_batch(x_batch, y_batch)
					i = 0
					x_batch = []
					y_batch = []

			except StopIteration as e:
					if len(x_batch) > 0:
						yield process_batch(x_batch, y_batch)
					self.reset()
					break

	def get_num_files(self):
		return self.generatorObj.get_num_files()

	def count_batches(self):
		"""
		Exhaust the generator to count the number of batches. Then reset.

		Returns:
			int: number of batches in the dataset
		"""
		i = 0
		while True:
			try:
				x,y = next(self.generator)
				i += 1
			except StopIteration as e:
				self.reset()
				return np.ceil(i / self.batch_size)


if __name__ == "__main__":

	source = ""
	b = LIDGenerator(source, 10, True, ["spanish", "english"])
	gen = b.get_generator()
	for audio, label, onehot in gen:
		print(audio)
		print(label)
		print(onehot)


	a = AudioGenerator(source, 10, shuffle=True, run_only_once=True)
	gen = a.get_generator()

	i = 0
	for data, fs, fn in gen:
		print(data)
		print(len(data))
		if i > 20:
			break
		else:
			# wav.write(filename=(fn[:-4]+str(i)+".wav"), rate=fs, data=data)
			i += 1
