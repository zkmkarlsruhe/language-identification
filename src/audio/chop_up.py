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


def chop_up_audio (file_name, desired_length_s = 5,
					min_length_s = 2.5, max_silence = 1.5,
					threshold = 60, padding = "Silence",
					audio_window_ms = 10):
	
	assert(min_length <= desired_length_s)

	sample_rate = auditok.WaveAudioSource(file_name).sampling_rate

	audio_window = audio_window_ms / 1000
	nn_input_len = desired_length_s * sample_rate

	# chop up the audio using auditok
	regions = auditok.split(file_name, min_dur=min_length, max_dur=desired_length_s, 
							max_silence=max_silence, strict_min_dur=True, 
							analysis_window=audio_window, energy_threshold=threshold)

	# extend tokens to desired length
	audio_cuttings = []
	for i, r in enumerate(regions):

		numpy_data = to_array(r._data, 2, 1)
		r.save("region_{meta.start:.3f}-{meta.end:.3f}.wav")

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
	parser.add_argument('--min_length', type=float, default=2.5,
						help='length of valid audio frames in seconds')
	parser.add_argument('--max_silence', type=float, default=1.5,
						help='maximum length of silent audio frames inside one token in seconds')
	parser.add_argument('--energy_threshold', type=float, default=60, 
						help='amount of energy that determines valid audio frames')
	# Neural Network Preprocessing
	padding_choices = ("Silence", "Data", "Noise")
	parser.add_argument('--padding', type=str, default="Data", choices=padding_choices,
						help='whether to pad extracted tokens with silence, data or noise')
	parser.add_argument('--audio_length_s', type=float, default=5,
						help='desired output length in seconds')
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
