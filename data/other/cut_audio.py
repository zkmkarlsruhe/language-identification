"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under Simplified BSD License.
"""

"""
A small script to ensure the audio files have the correct size
"""

import os
import glob
import argparse
import threading
import numpy as np
from queue import Queue

import scipy.io.wavfile as wav

from src.audio.utils import pad_with_silence


def task(filepath, output_dir, target_length_s, minimum_length = 0.2):

	# load the wav
	fs, audio = wav.read(filepath)
	file_name = filepath.split('/')[-1]

	# cut
	target_length = target_length_s * fs
	num_segments = int(np.ceil(len(audio) / target_length))

	# for all segments create slices of target length
	for i in range(0, num_segments):
		slice_start = int(i * target_length)
		slice_end = int(slice_start + target_length)
		rest = len(audio) - slice_start

		# if we have only one segment left and there is at least
		#  minimum_length data pad it with silence
		if i == num_segments-1:
			if rest >= target_length * minimum_length:
				chunk = pad_with_silence(audio[slice_start:], target_length)
			else:
				break
		else:
			chunk = audio[slice_start:slice_end]

		# save
		chunk_name = file_name[:-3] + str(i) + ".wav"
		chunk_path = os.path.join(output_dir, chunk_name)
		wav.write(chunk_path, fs, chunk)


def taskEnclosures(i, q):
	while True:
		filepath, output_dir, target_length_s = q.get()
		task(filepath, output_dir, target_length_s)
		q.task_done()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str,
						required=True,
						help="path to the audio directory")
	parser.add_argument('--output_dir', type=str,
						required=True,
						help="path to the output directory")
	parser.add_argument('--target_length_s', type=int,
						default=5,
						help="target length of the processed audio file in seconds")
	parser.add_argument('--num_threads', type=int,
						default=4,
						help="amount of worker threads")
	args = parser.parse_args()

	# create output dir
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# worker queue
	queue = Queue()

	# create workers
	for i in range(args.num_threads):
		worker = threading.Thread(target=taskEnclosures, args=(i, queue,), daemon=True)
		worker.start()

	# search input dir for wavs and put the arguments on the queue
	filenames = glob.glob(os.path.join(args.input_dir, '*.wav'))
	for filename in filenames:
		fn_args = (filename, args.output_dir, args.target_length_s)
		queue.put(fn_args)

	# wait until the workers are ready
	queue.join()
