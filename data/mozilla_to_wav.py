"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under Simplified BSD License.
"""

import os
import pydub
import argparse
import threading
import numpy as np
import scipy.io.wavfile as wav

import shutil
from yaml import load
from src.audio.chop_up import chop_up_audio

def sentence_is_too_short(sentence_len, language):
	if language == "mandarin":
		return sentence_len < 3
	else:
		return sentence_len < 6


def traverse_csv(language, input_dir, output_dir, max_chops, 
				desired_audio_length_s, sample_rate, sample_width,
				allowed_downvotes, remove_raw, min_length_s, max_silence_s,
				energy_threshold, use_validated_set):
	"""
	travers the language specific file, extract and save important samples.
	"""
	
	lang = language["lang"]
	lang_abb = language["dir"]

	input_sub_dir = os.path.join(input_dir, lang_abb)
	input_sub_dir_clips = os.path.join(input_sub_dir, "clips")


	splits = ["train", "dev", "test"]

	fast_forward = 0	

	for split_index, split in enumerate(splits):

		output_dir_wav = os.path.join(output_dir, "wav", split, lang)
		output_dir_raw = os.path.join(output_dir, "raw", split, lang)

		# create subdirectories in the output directory
		if not os.path.exists(output_dir_wav):
			os.makedirs(output_dir_wav)
		if not os.path.exists(output_dir_raw):
			os.makedirs(output_dir_raw)

		# keep track of files handled
		processed_files = 0
		produced_files = 0
		to_produce = int(max_chops[split_index])
		done = False

		if use_validated_set:
			input_clips_file = os.path.join(input_sub_dir, "validated.tsv")
			if to_produce == -1: 
				print("when using validated.tsv, please set number of chops to a positive number")
				exit()
		else:
			input_clips_file = os.path.join(input_sub_dir, split + ".tsv")


		# open mozillas' dataset file
		with open(input_clips_file) as f:

			try:
				# skip the first line
				line = f.readline()

				# when using the validated.tsv we have to start where we left off
				if use_validated_set:
					for skip in range(fast_forward):
						f.readline()

				while True:

					# get a line
					line = f.readline().split('\t')

					# if the line is not empty
					if line[0] != "":

						# check if the sample contains more than X symbols 
						# and has not more than Y downvotes
						sentence = line[2]
						too_short = sentence_is_too_short(len(sentence), language["lang"])
						messy = int(line[4]) > allowed_downvotes
						if too_short or messy:
							continue

						# get mp3 filename
						mp3_filename = line[1]
						mp3_path = os.path.join(input_sub_dir_clips, mp3_filename)
						wav_path_raw = os.path.join(output_dir_raw,
													mp3_filename[:-4] + ".wav")

						# convert mp3 to wav
						audio = pydub.AudioSegment.from_mp3(mp3_path)
						audio = pydub.effects.normalize(audio)
						audio = audio.set_frame_rate(sample_rate)
						audio = audio.set_channels(1)
						audio = audio.set_sample_width(sample_width)
						audio.export(wav_path_raw, format="wav")
						processed_files += 1

						# chop up the samples and write to file
						rand_int = np.random.randint(low=0, high=2)
						padding_choice = ["Data", "Silence"][rand_int]
						chips = chop_up_audio (wav_path_raw, padding=padding_choice,
											desired_length_s=desired_audio_length_s,
											min_length_s=min_length_s, max_silence_s=max_silence_s,
											threshold=energy_threshold)
						for chip_name, chip_fs, chip_data in chips:
							wav_path = os.path.join(output_dir_wav, chip_name + ".wav")
							wav.write(wav_path, chip_fs, chip_data)
							produced_files += 1
							# remove the intermediate file
							if remove_raw and os.path.exists(wav_path_raw):
								os.remove(wav_path_raw)
							# check if we are done yet
							if to_produce != -1 and produced_files >= to_produce:
								done = True
								break

						if done:
							# when using the validated.tsv we have to make sure the same
							# speakers wont appear in more than one set. Fast-forward...
							if use_validated_set:
								last_speaker = speaker = line[0]
								while speaker == last_speaker:
									speaker = f.readline().split('\t')[0]
									fast_forward += 1
							break

					else:
						print("Nothing left!")
						break

			except Exception as e:
				print("Error:", e)

			print("Processed %d mp3 files for %s-%s" % (processed_files, lang, split))
			print("Produced  %d wav files for %s-%s" % (produced_files, lang, split))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--config_path', default=None,
						help="path to the config yaml file. When given, arguments will be ignored")
	parser.add_argument("--cv_dir", type=str, default=None,
						help="directory containing all languages")
	parser.add_argument("--cv_filtered_dir", type=str, default="../res",
						help="directory to receive converted clips of all languages")
	# Data 
	parser.add_argument("--max_chops", type=int, nargs=3, default=[-1, -1, -1],
						help="amount of maximum wav chops to be produced per split. -1 means all.")
	parser.add_argument("--use_validated_set", type=bool, nargs=3, default=False,
						help="whether to use the train, test and dev sets or all validated data")
	parser.add_argument("--allowed_downvotes", type=int, default=0,
						help="amount of downvotes allowed")
	# Audio file properties
	parser.add_argument("--audio_length_s", type=int, default=10,
						help="length of wav files being produced")
	parser.add_argument("--min_length_s", type=float, default=2.5,
						help="min length of an audio event")
	parser.add_argument("--max_silence_s", type=float, default=1,
						help="max length of silence in an audio event")
	parser.add_argument("--energy_threshold", type=float, default=60,
						help="minimum energy for a frame to be valid")
	parser.add_argument("--sample_rate", type=int, default=16000,
						help="sample rate of files being produced")
	parser.add_argument('--sample_width', type=int, default=2, choices=(1, 2, 4),
						help='number of bytes per sample')
	parser.add_argument("--use_random_padding", type=bool, default=True,
						help="whether to randomly use silence or data padding")
	# System
	parser.add_argument("--parallelize", type=bool, default=True,
						help="whether to use multiprocessing")
	parser.add_argument("--remove_raw", type=bool, default=True,
						help="whether to remove intermediate file")
	args = parser.parse_args()

	# overwrite arguments when config is given
	if args.config_path:
		config = load(open(args.config_path, "rb"))
		if config is None:
				print("Could not find config file")
				exit(-1)
		else:
			args.cv_dir            = config["cv_dir"]
			args.cv_filtered_dir   = config["cv_filtered_dir"]
			args.max_chops         = config["max_chops"]
			args.allowed_downvotes = config["allowed_downvotes"]
			args.audio_length_s    = config["audio_length_s"] 
			args.max_silence_s     = config["max_silence_s"] 
			args.min_length_s      = config["min_length_s"] 
			args.energy_threshold  = config["energy_threshold"] 
			args.sample_rate       = config["sample_rate"]
			args.sample_width      = config["sample_width"]
			args.parallelize       = config["parallelize"]
			args.remove_raw        = config["remove_raw"]
			args.use_validated_set = config["use_validated_set"]
			language_table         = config["language_table"]

			# copy config to output dir
			if not os.path.exists(args.cv_filtered_dir):
				os.makedirs(args.cv_filtered_dir)
			shutil.copy(args.config_path, args.cv_filtered_dir)

	else:
		language_table = [
			{"lang": "english", "dir": "en"},
			{"lang": "german", "dir": "de"},
			{"lang": "french", "dir": "fr"},
			{"lang": "spanish", "dir": "es"},
			{"lang": "mandarin", "dir": "zh-CN"},
			{"lang": "russian", "dir": "ru"},
			# {"lang": "unknown", "dir": "ja"},
			# {"lang": "unknown", "dir": "ar"},
			# {"lang": "unknown", "dir": "ta"},
			# {"lang": "unknown", "dir": "pt"},
			# {"lang": "unknown", "dir": "tr"},
			# {"lang": "unknown", "dir": "it"},
			# {"lang": "unknown", "dir": "uk"},
			# {"lang": "unknown", "dir": "el"},
			# {"lang": "unknown", "dir": "id"},
			# {"lang": "unknown", "dir": "fy-NL"},
		]

	# count the number of unknown languages
	# unknown = 0
	# for language in languages:
	#     if language["lang"] == "unknown":
	#         unknown += 1
	# if unknown > 0:
	#     number_unknown = args.number // unknown

	threads = []
	for language in language_table:

		# clips_per_language = args.max_chops
		# if language["lang"] == "unknown":
		#     clips_per_language = number_unknown

		# prepare arguments
		function_args = (language, args.cv_dir, args.cv_filtered_dir, args.max_chops, 
						args.audio_length_s, args.sample_rate, args.sample_width, 
						args.allowed_downvotes, args.remove_raw, args.min_length_s,
						args.max_silence_s, args.energy_threshold, args.use_validated_set)

		# process current language for all splits
		if args.parallelize:
			threads.append(threading.Thread(target=traverse_csv, args=function_args,
											daemon=True) )
		else:
			traverse_csv(*function_args)

	# wait for threads to end
	if args.parallelize:
		for t in threads:
			t.start()
		for t in threads:
			t.join()

	if args.remove_raw:
		shutil.rmtree(os.path.join(args.cv_filtered_dir, "raw"))
