"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under Simplified BSD License.
"""

import nlpaug.flow as flow
import nlpaug.augmenter.audio as naa

from .utils import pad_with_silence


class AudioAugmenter(object):
	def __init__(self, target_length_s, fs):
		self.fs = fs
		self.target_length_s = target_length_s
		shift = naa.ShiftAug(sampling_rate=fs, direction='random', duration=0.2)
		crop = naa.CropAug(sampling_rate=fs, zone=(0.2, 0.8), coverage=0.02)
		vltp = naa.VtlpAug(sampling_rate=fs, zone=(0.2, 0.8), coverage=0.8, 
							fhi=4800, factor=(0.9, 1.1))
		noise = naa.NoiseAug(zone=(0.0, 1.0), coverage=1, color='white')
		speed = naa.SpeedAug(zone=(0.0, 1.0), coverage=0.1, factor=(0.9, 1.1))
		pitch = naa.PitchAug(sampling_rate=16000, zone=(0, 1), coverage=0.3, factor=(0, 2.1))
		# comment the augmentation that you don't need
		self._aug_flow = flow.Sequential([
			shift,
			crop,
			vltp,
			speed,
			pitch,
			noise,
		])

	def augment_audio_array(self, signal):
		"""Augment a single or list of audio signals

		Args:
			signal (array or list): signal or list of signals

		Returns:
			[list]: a list of augmented signals padded with silence 
		"""
		augmented_data = self._aug_flow.augment(signal, num_thread=8)
		data = []
		for x in augmented_data:
			x = pad_with_silence(x, self.target_length_s * self.fs)
			data.append(x)
		return data

	def augment_audio(self, signal):
		"""Augment a single audio signal

		Args:
			signal (array): signal 

		Returns:
			[array]: augmented signal padded with silence
		"""
		augmented_data = self._aug_flow.augment(signal)
		data = pad_with_silence(augmented_data, self.target_length_s * self.fs)
		return data


if __name__ == "__main__":

	import matplotlib.pyplot as plt
	import scipy.io.wavfile as wav

	file_name = ""
	fs, arr = wav.read(filename=file_name)
	a = AudioAugmenter(fs)

	runs = 4
	plt.figure()
	plt.subplot(runs+1, 1, 1)
	plt.plot(arr)
	for i in range(runs):
		plt.subplot(runs+1, 1, i+2)
		data = a.augment_audio_array(arr, fs)
		plt.plot(data)
	plt.tight_layout()
	plt.show()
