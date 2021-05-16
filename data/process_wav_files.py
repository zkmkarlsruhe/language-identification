"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under GNU GPL Version 3.
"""

import os
import argparse
import threading
import shutil

import scipy.io.wavfile as wav
from imageio import imwrite
from yaml import load
from audio.generators import AudioGenerator, pad_with_silence
from audio.augment import AudioAugmenter
from audio.features import signal_to_features, normalize


def process_audio_dir(lang, wav_dir, aug_dir = None, img_dir = None,
				audio_length_s = 10, augment_nu = 5, feature_type = "mfcc",
				frame_size_ms = "20", feature_nu = 12):

    # create dirs
    if aug_dir:
        if not os.path.exists(aug_dir):
            os.makedirs(aug_dir)
    if img_dir:
		if not os.path.exists(img_dir):
        	os.makedirs(img_dir)

    # create generator
    generator = AudioGenerator(source=os.path.join(wav_dir, lang),
                                target_length_s=audio_length_s,
                                dtype="float32",
                                shuffle=True, run_only_once=True)
    generator_queue = generator.get_generator()

    # generate and process audio chunks
    i = 0
    while True:
        try:
            # receive an audio block of desired length and normalize it
            data, fs, file_name = next(generator_queue)
            data = normalize(data)
            data_list = [data]
            file_name = os.path.split(file_name)[-1]

            # augment the data chunks
            if augment_nu > 0:
                augmenter = AudioAugmenter(fs)
                for k in range(augment_nu):
                    data_aug = augmenter.augment_audio_array(signal=data, fs=fs)
                    data_aug = pad_with_silence(data_aug, audio_length_s * fs)
                    data_aug = normalize(data_aug)
                    data_list.append(data_aug)

            # save audio chunks as wav files
            if aug_dir:
                for k, audio in enumerate(data_list):
                    if k == 0:
                        file_name_temp = os.path.join(aug_dir, file_name[:-4] + "{}".format(i) +".wav")
                    else:
                        file_name_temp = os.path.join(aug_dir, file_name[:-4] + "{}_aug{}".format(i, k) + ".wav")
                    wav.write(file_name_temp, fs, audio)

            # for all chunks compute features and save images to target directory
            if img_dir:
                for k, data_entry in enumerate(data_list):
                    feature_array = signal_to_features(signal=data_entry, fs=fs,
                                                        len_segment_ms=frame_size_ms,
                                                        num_features=feature_nu,
                                                        audio_feature=feature_type,
                                                        zero_center=True,
                                                        remap=(0, 1))
                    file_name_feat = os.path.join(img_dir, file_name[:-4] +"_{}_{}".format(i, k) + ".png")
                    feature_array *= 255
                    imwrite(file_name_feat, feature_array.astype("uint8"))

            i += 1

            if i % 1000 == 0:
                print("Processed {} audio chunks".format(i))

        except StopIteration:
            print("language '{}' Stopped on {}".format(lang, i))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config_path', default=None,
                        help="path to the config yaml file. When given, arguments will be ignored")
    parser.add_argument('--wav_dir', type=str, default=None,
                        help="directory to search for language folders")
    parser.add_argument('--audio_length_s', type=int, default=10,
                        help="length of the audio window to process in seconds")
    parser.add_argument('--languages', type=str, nargs='+',default=None,
                        help="languages to process")
    parser.add_argument("--parallelize_pro", action="store_true", default=False,
                        help="whether to use multiprocessing")
    # Augment arguments
    parser.add_argument('--aug_dir', type=str, default=None,
                        help="directory to save processed wav files")
    parser.add_argument('--augment_nu', type=int, default=3,
                        help="number of augmentations to do per file found")
    # Feature arguments
    parser.add_argument('--img_dir', type=str, default=None,
                        help="directory to save audio features as .png files to")
    parser.add_argument('--feature_type', choices=("mfcc", "logfbank", "logfbankenergy", "spectrogram"),
                        default="logfbankenergy",
                        help="type of feature to compute: one of mfcc, logfbank, logfbankenergy and spectrogram")
    parser.add_argument('--feature_frame_size_ms', type=int, default=20,
                        help="length of an audio frame to compute features from in milliseconds")
    parser.add_argument('--feature_nu', type=int, default=40,
                        help="amount of features to compute per audio frame")
    args = parser.parse_args()

    # overwrite arguments when config is given
    if args.config_path:
        config = load(open(args.config_path, "rb"))
        if config is None:
                print("Could not find config file")
                exit()
        args.wav_dir         = config["wav_dir"]
        args.audio_length_s  = config["audio_length_s"]
        args.parallelize_pro = config["parallelize_pro"]
        args.aug_dir         = config["aug_dir"]
        args.augment_nu      = config["augment_nu"]
        args.img_dir         = config["img_dir"]
        args.feature_type    = config["feature_type"]
        args.feature_nu      = config["feature_nu"]
        args.languages       = config["languages"]

        # copy config to output dirs
        if args.aug_dir:
            if not os.path.exists(args.aug_dir):
                os.makedirs(args.aug_dir)
            shutil.copy(args.config_path, args.aug_dir)
        if args.img_dir:
            if not os.path.exists(args.img_dir):
                os.makedirs(args.img_dir)
            shutil.copy(args.config_path, args.img_dir)

    # assertions
    if args.wav_dir == None:
        print("No wave source dir provided!")
        exit()
    if args.img_dir == None:
        print("Please provide an output dir")
        exit()

    splits = ["train", "dev", "test"]
    for split in splits:
        
        # append paths with current split 
        wav_dir = os.path.join(args.wav_dir, split)
        img_dir = os.path.join(args.img_dir, split)

        # augmentation is only ment for training purposes
        if split == "train":
            aug_dir = args.aug_dir
            augment_nu = args.augment_nu
        else: 
            aug_dir = None
            augment_nu = 0
        
        threads = []
        for language in args.languages:

            # append paths with current language
            aug_lang_dir = None
            if aug_dir:
                aug_lang_dir = os.path.join(aug_dir, language)
            img_lang_dir = os.path.join(img_dir, language)

            # prepare arguments
            function_args = (language, wav_dir, aug_lang_dir, img_lang_dir,
                            args.audio_length_s, augment_nu, args.feature_type,
                            args.feature_frame_size_ms, args.feature_nu)
            
            # process current language for current split
            if args.parallelize_pro:
                threads.append( threading.Thread(target=process_audio_dir, args=function_args,
                                                daemon=True) )
            else:
                process_audio_dir(*function_args)

        # wait for threads to end
        if args.parallelize_pro:
            for t in threads:
                t.start()
            for t in threads:
                t.join()
