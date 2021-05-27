"""
A small script for downloading audio files from the "audioset" dataset using youtube-dl and ffmpeg
The list of files is passed as unbalanced_train_segments.csv
Unfortunately, the balanced_train_segments.csv has a slightly different style
Labels can be ignored by adding them to the restrictions list
"""

import os
import argparse
import threading
import subprocess
from Queue import Queue



def downloadEnclosures(i, q):
    while True:
        yt_url, start_s, length_s, output_dir = q.get()
        download(yt_url, start_s, length_s, output_dir)
        q.task_done()


def download(yt_url, start_s, length_s, output_dir):
	command = """youtube-dl {} --extract-audio --audio-format wav -o "{}/%(title)s.%(ext)s" --postprocessor-args '-ss {} -t {} -ac 1 -ar 16000' """.format(yt_url, output_dir, start_s, length_s)
	subprocess.call(command, shell=True)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input_file', type=str,
						required=True,
						help="path to the unbalanced_train_segments.csv")
	parser.add_argument('--output_dir', type=str,
						default=os.path.join(os.getcwd(), "yt-noise"),
						help="path to the output directory")
	parser.add_argument('--num_threads', type=int,
						default=4,
						help="amount of worker threads")
	parser.add_argument('--downloads', type=int,
						default=100000,
						help="amount of audio pieces to download")
	args = parser.parse_args()

	# worker queue
	queue = Queue()

	# create workers
	for i in range(args.num_threads):
		worker = threading.Thread(target=downloadEnclosures, args=(i, queue,), daemon=True)
		worker.start()

	# labels of audioset that should not be considered "noise"
	# for example: human speech, singing
	# labels taken from ontology.json ()
	restrictions = [

		### human sounds ###
			# "/m/09l8g", # Human voice
				"/m/09x0r", # Speech
					"/m/05zppz", "/m/02zsn", "/m/0ytgt", "/m/01h8n0", "/m/02qldy", "/m/0261r1", "/m/0brhx",
				"/m/02rtxlg", # whispering
				"/m/015lz1", # singing
					"/m/0l14jd", "/m/01swy6", "/m/02bk07", "/t/dd00003", "/t/dd00004", "/t/dd00005", "/t/dd00006", "/m/06bxc",

		#### Music ###
			"/m/05lls", # opera
			"/m/0y4f8", # vocal music
				"/m/0z9c", "/m/02cz_7",
	]

	# labels of audioset that should be considered "noise"
	# for example: laughing, human locomotion
	positives = [

		### human sounds ###

		"/m/0bpl036", # human locomotion
			"/m/06h7j", "/m/07qv_x_", "/m/07pbtc8",
		"/m/0k65p", # hands
			"/m/025_jnm", "/m/0l15bq",
		"/t/dd00012", # human group actions
			"/m/0l15bq", # clapping
			"/m/053hz1", # cheering
			"/m/028ght", # applause
			"/m/07rkbfh", # chatter
			"/m/03qtwd", # crowd
			"/m/07qfr4h", # speech noise
			"/m/04v5dt", # Booing
			"/t/dd00013", # Children playing
			"/t/dd00135", # Children shouting
		"/m/0463cq4",  # crying
			"/t/dd00002", "/m/07qz6j3",
		"/m/02fxyj", # Humming
		"/m/07s2xch", # Groan
		"/m/07r4k75", # Grunt
		"/m/01j423", # Yawn
		"/m/07qw_06", # wail, moan
		"/m/07plz5l", # sigh
		"/m/01w250", # whistling
		"/m/09hlz4", # respiratory sounds
			"/m/0lyf6", # breathing
				"/m/07mzm6", "/m/01d3sd", "/m/07s0dtb", "/m/07pyy8b", "/m/07q0yl5",
			"/m/01hsr_", # sneeze
			"/m/07ppn3j", # sniff
			"/m/01b_21", # cough
				"/m/0dl9sf8", # throat clearing
		"/m/07p6fty", # shout
			"/m/07q4ntr", # bellow
			"/m/07rwj3x", # whoop
			"/m/07sr1lc", # yell
			"/m/04gy_2", # battle_cry
			"/t/dd00135", # children shouting
		"/m/03qc9zr", # screaming
		"/m/01j3sz", # laughter
			"/t/dd00001", "/m/07r660_", "/m/07s04w4", "/m/07sq110", "/m/07rgt08",
		"/m/02p3nc"  # hiccup
	]

	# open Youtube's dataset file
	with open(args.input_file) as f:

		# run for a certain number of downloads
		num_files = args.downloads
		file_count = 0

		try:
			#  skip the first three line
			print(f.readline())
			print(f.readline())
			print(f.readline())

			# as long as we didn't reach the maximum number of files
			while file_count < num_files:

				# get a line
				line = f.readline()[:-1].split(',')
				# print(line)

				# if the line is not empty
				if line[0] != "":

					# get the URL and start and end points
					URL = "https://www.youtube.com/watch?v=" + line[0]
					start = float(line[1])
					end = float(line[2])
					audio_length = end - start

					# get the labels from csv and clean them up
					labels = []
					rest = line[3:]
					for i, label in enumerate(rest):
						if i == 0:
							label = label[2:]
						if i == len(rest)-1:
							label = label[:-1]
						labels.append(label)

					# if audio_length != 10.0:
					# 	print("Sample not 10 seconds")
					# 	continue

					# apply label restrictions
					if any(label in labels for label in restrictions):
						print("Found restricted label in {}".format(labels))
						continue

					if not any (label in labels for label in positives):
						print("Label not in positives!")
						continue

					print("Something in {} is important and nothing restricted". format(labels))

					# get the data and save it
					try:
						function_args = (URL, start, audio_length, args.output_dir)
						queue.put(function_args)
					except Exception as e:
						print("Download oopsi: ", e)
						continue

					file_count += 1

				else:
					print("Nothing left!")
					break

		except EOFError as e:
			print("End of file!")
