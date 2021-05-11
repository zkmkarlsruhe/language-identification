"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under GNU GPL Version 3.
"""

import argparse
import os
import time

import auditok
from utils import pad_with_data, pad_with_noise, pad_with_silence, to_array


def chop_up_audio (file_name, desired_length_s = 10,
					min_length = 250, max_silence = 150,
					threshold = 60, padding = "Silence",
					audio_window_ms = 10):
	
	sample_rate = auditok.WaveAudioSource(file_name).sampling_rate
	min_length  *= audio_window_ms / 1000
	max_length  = desired_length_s / audio_window_ms * 1000
	max_silence *= audio_window_ms / 1000
	audio_window = audio_window_ms / 1000
	nn_input_len = desired_length_s * sample_rate

	assert(min_length <= max_length)

	# chop up the audio using auditok
	regions = auditok.split(file_name, min_dur=min_length, max_dur=max_length, 
							max_silence=max_silence, strict_min_dur=True, 
							analysis_window=audio_window, energy_threshold=threshold)

	# extend tokens to desired length
	audio_cuttings = []
	for i, r in enumerate(regions):

		numpy_data = to_array(r._data, 2, 1)

		if padding == "Silence":
			extended_token = pad_with_silence(numpy_data, nn_input_len)
		elif padding == "Data":
			extended_token = pad_with_data(numpy_data, nn_input_len)
		else:
			extended_token = pad_with_noise(numpy_data, nn_input_len)

		file_name_out = os.path.split(file_name)[-1][:-4] + "_" + str(i)
		data_tuple = (file_name_out, sample_rate, extended_token)
		audio_cuttings.append(data_tuple)

	return audio_cuttings


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Remove silence form wav-files')
	# Audio Source
	parser.add_argument('--file_name', type=str, required= True,
						help='file name to read from')
	parser.add_argument('--audio_window_ms', type=int, default=10,
						help='length of an audio frame in milliseconds')
	# Tokenization
	parser.add_argument('--min_length', type=int, default=250,
						help='minimum number of valid audio frames')
	parser.add_argument('--max_silence', type=int, default=150,
						help='maximum number of silent audio frames inside one token')
	parser.add_argument('--energy_threshold', type=float, default=60, 
						help='amount of energy that determines valid audio frames')
	# Neural Network Preprocessing
	padding_choices = ("Silence", "Data", "Noise")
	parser.add_argument('--padding', type=str, default="Data", choices=padding_choices,
						help='whether to pad extracted tokens with silence, data or noise')
	parser.add_argument('--audio_length_s', type=int, default=10,
						help='desired output length')
	# Other
	parser.add_argument('--output_dir', type=str, default="./",
						help='directory to store wav files to')

	args = parser.parse_args()

	# sampling parameters
	audio_window_ms = args.audio_window_ms
	file_name = args.file_name

	# buffer parameters
	threshold = args.energy_threshold

	# tokenization parameters
	min_length = args.min_length
	max_silence = args.max_silence
	# desired length should not be smaller than the max length of a token

	audio_length_s = args.audio_length_s
	padding = args.padding

	# others
	output_dir = args.output_dir

	chunk = chop_up_audio(file_name, audio_length_s, min_length, max_silence,
							threshold, padding, audio_window_ms)
	
	import scipy.io.wavfile as wav

	for item in chunk:
		file_path = os.path.join(output_dir, item[0] + ".wav")
		wav.write(file_path, item[1], item[2])
