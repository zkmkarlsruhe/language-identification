"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under Simplified BSD License.
"""

import os
import glob
from shutil import copyfile, move
import argparse


def split_dataset_to_cv_dir(src, dest, split, target_name, move_it=True):

	file_paths = glob.glob(os.path.join(src, '*.wav'))
	num_files = len(file_paths)

	for i,file_path in enumerate(file_paths):
		if i / num_files <= split[0]:
			dest_split = "train"
		elif i / num_files <= split[0] + split[1]:
			dest_split = "test"
		else:
			dest_split = "dev"
		
		dest_dir = os.path.join(dest, dest_split, target_name)

		# create output dir
		if not os.path.exists(dest_dir):
			os.makedirs(dest_dir)

		file_name = file_path.split('/')[-1]
		new_file_path = os.path.join(dest_dir, file_name)
		if move_it:
			move(file_path, new_file_path)
		else:
			copyfile(file_path, new_file_path)

	if move:
		print("moved " + str(i) + " files")
	else:
		print("copyied " + str(i) + " files")


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', required=True, help="noise directory")
	parser.add_argument('--output_dir', required=True, help="common voice root directory")
	parser.add_argument('--target_name', default="__noise", help="class and directory name")
	parser.add_argument('--ratio', type=float, nargs=3, default=[0.8,0.1,0.1], help="put -1 for all")
	parser.add_argument('--move', type=bool, default=True, help="whether to move or just copy")
	cli_args = parser.parse_args()
	split_sum = cli_args.ratio[0] + cli_args.ratio[1] + cli_args.ratio[2]
	assert(split_sum == 1.0)
	split_dataset_to_cv_dir(cli_args.input_dir, cli_args.output_dir, cli_args.ratio, cli_args.target_name, cli_args.move)
