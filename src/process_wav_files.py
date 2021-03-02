import os
import argparse

import scipy.io.wavfile as wav
from generators import AudioGenerator, pad_with_silence
from audio_augment import AudioAugmenter
from audio_features import signal_to_features, normalize

from imageio import imwrite


def process_audio_dirs(args, categories):

    # get params
    source_dir = args.source
    target_length_s = args.target_length_s

    wav_dir = args.wav_dir
    augment = args.augment
    augment_nu = args.augment_nu

    img_dir = args.img_dir
    feature_type = args.feature_type
    frame_size_ms = args.feature_frame_size_ms
    feature_nu = args.feature_nu

    # create generators
    generators = [AudioGenerator(source=os.path.join(source_dir, category),
                                 target_length_s=target_length_s,
                                 dtype="float32",
                                 shuffle=True, run_only_once=True)
                  for category in categories]
    generator_queues = [audioGen.get_generator() for audioGen in generators]

    # create dirs
    for category in categories:
        if wav_dir:
            temp_dir = os.path.join(wav_dir, category)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
        if img_dir:
            output_dir = os.path.join(img_dir, category)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

    # generate and process audio chunks
    i = 0
    while True:
        try:
            # iterate over all directories
            for j, category in enumerate(categories):

                # receive an audio block of desired length and normalize it
                data, fs, file_name = next(generator_queues[j])
                data = normalize(data)
                data_list = [data]

                # augment the data chunks
                if augment:
                    augmenter = AudioAugmenter(fs)
                    for k in range(augment_nu):
                        data_aug = augmenter.augment_audio_array(signal=data, fs=fs)
                        data_aug = pad_with_silence(data_aug, target_length_s * fs)
                        data_list.append(data_aug)

                # save audio chunks as wav files
                if wav_dir:
                    for k, audio in enumerate(data_list):
                        if k == 0:
                            file_name_temp = os.path.join(wav_dir, category, file_name[:-4] + "{}".format(i) +".wav")
                        else:
                            file_name_temp = os.path.join(wav_dir, category, file_name[:-4] + "{}_aug{}".format(i, k) + ".wav")
                        wav.write(file_name_temp, fs, audio)

                # for all chunks compute features and save images to target directory
                if img_dir:
                    for k, data_entry in enumerate(data_list):
                        feature_array = signal_to_features(signal=data_entry, fs=fs,
                                                           len_segment_ms=frame_size_ms,
                                                           num_features=feature_nu,
                                                           audio_feature=feature_type,
                                                           zero_center=True,
                                                           normalize=(0, 1))
                        file_name_feat = os.path.join(img_dir, category, file_name[:-4] +"_{}_{}".format(i, k) + ".png")
                        feature_array *= 255
                        imwrite(file_name_feat, feature_array.astype("uint8"))

                i += 1

            if i % 1000 == 0:
                print("Processed {} audio chunks".format(i))

        except StopIteration:
            # TODO give option for balancing
            print("Category '{}' Stopped on {}".format(category, i))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--source', type=str,
                        required=True,
                        help="directory to search for language folders")
    parser.add_argument('--target_length_s', type=int, default=10,
                        help="length of the audio window to process in seconds")

    parser.add_argument('--wav_dir', type=str,
                        help="directory to save processed wav files")
    parser.add_argument('--augment', action='store_true', default=False,
                        help="whether to augment the data")
    parser.add_argument('--augment_nu', type=int, default=6,
                        help="number of augmentations to do per file found")

    parser.add_argument('--img_dir', type=str,
                        help="directory to save audio features as .png files to")
    parser.add_argument('--feature_type', choices=("mfcc", "logfbank", "logfbankenergy", "spectrogram"),
                        default="logfbankenergy",
                        help="type of feature to compute: one of mfcc, logfbank, logfbankenergy and spectrogram")
    parser.add_argument('--feature_frame_size_ms', type=int, default=20,
                        help="length of an audio frame to compute features from in milliseconds")
    parser.add_argument('--feature_nu', type=int, default=40,
                        help="amount of features to compute per audio frame")

    cli_args = parser.parse_args()

    # Start a spectrogram generator for each class
    # Each generator will scan a directory for audio files and convert them to images
    # adjust this if you have other categories or any category is missing
    categories = [
        "russian",
        "english",
        "german",
        "french",
        "spanish",
    ]

    process_audio_dirs(cli_args, categories)
