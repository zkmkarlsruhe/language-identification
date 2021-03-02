import os
import random
from shutil import copyfile
import argparse


def copy_list_of_files(dir_path_from, dir_path_to, file_list):
    if not os.path.exists(dir_path_to):
        os.makedirs(dir_path_to)

    for file in file_list:
        fr = os.path.join(dir_path_from, file)
        to = os.path.join(dir_path_to, file)
        copyfile(fr, to)


def create_train_val_test(path, new_path):

    for child in os.listdir(path):

        subdir_path = os.path.join(path, child)

        # check if entry is a directory
        if os.path.isdir(subdir_path):
            # check for files
            subdir_list = os.listdir(subdir_path)
            total = len(subdir_list)
            random.shuffle(subdir_list)

            train_path = os.path.join(new_path, "train", child)
            val_path = os.path.join(new_path, "val", child)
            test_path = os.path.join(new_path, "test", child)

            copy_list_of_files(subdir_path, train_path, subdir_list[:total * 7 // 10])
            copy_list_of_files(subdir_path, val_path, subdir_list[total * 7 // 10:total * 9 // 10])
            copy_list_of_files(subdir_path, test_path, subdir_list[total * 9 // 10:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    cli_args = parser.parse_args()
    create_train_val_test(cli_args.source, cli_args.target)
