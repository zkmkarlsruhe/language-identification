"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under Simplified BSD License.
"""

import os
from shutil import copyfile, move
import argparse


def create_small_dataset(path, new_path, dataset_size):

	parent_list = os.listdir(path)
	for child in parent_list:

		subdir_path = os.path.join(path, child)
		new_subdir_path = os.path.join(new_path, child)

		# check if entry is a directory
		if os.path.isdir(subdir_path):

			# create new directory
			if not os.path.exists(new_subdir_path):
				os.makedirs(new_subdir_path)

			# check for files
			subdir_list = os.listdir(subdir_path)
			count = 0
			for file in subdir_list:
				if dataset_size != -1 or count < dataset_size:
					file_path = os.path.join(subdir_path, file)
					new_file_path = os.path.join(new_subdir_path, file)
					copyfile(file_path, new_file_path)
					# print(file_path, new_file_path)
				else:
					break
				count = count + 1
	
	print("copyied or moved" + str(count) + "files")


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--source', required=True)
	parser.add_argument('--target', required=True)
	parser.add_argument('--size', type=int, default=10, help="put -1 for all")
	cli_args = parser.parse_args()
	create_small_dataset(cli_args.source, cli_args.target, cli_args.size)
