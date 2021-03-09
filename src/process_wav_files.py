import os
import argparse
import threading

import scipy.io.wavfile as wav
from imageio import imwrite

from audio.generators import AudioGenerator, pad_with_silence
from audio.augment import AudioAugmenter
from audio.features import signal_to_features, normalize


def process_audio_dir(
    language,
    source_dir,
    wav_dir = None,
    img_dir = None,
    audio_length_s = 10,
    augment = True,
    augment_nu = 5,
    feature_type = "mfcc",
    frame_size_ms = "20",
    feature_nu = 12):


    # create generators
    generator = AudioGenerator(source=os.path.join(source_dir, language),
                                target_length_s=audio_length_s,
                                dtype="float32",
                                shuffle=True, run_only_once=True)

    generator_queue = generator.get_generator()

    print(wav_dir)
    print(img_dir)

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
            if wav_dir:
                for k, audio in enumerate(data_list):
                    if k == 0:
                        file_name_temp = os.path.join(wav_dir, file_name[:-4] + "{}".format(i) +".wav")
                    else:
                        file_name_temp = os.path.join(wav_dir, file_name[:-4] + "{}_aug{}".format(i, k) + ".wav")
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
            print("language '{}' Stopped on {}".format(language, i))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--source', type=str,
                        required=True,
                        help="directory to search for language folders")
    parser.add_argument('--audio_length_s', type=int, default=10,
                        help="length of the audio window to process in seconds")
    parser.add_argument("--parallelize_processes", action="store_true", default=False,
                        help="whether to use multiprocessing")
    # Augment arguments
    parser.add_argument('--wav_dir', type=str, default=None,
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

    parser.add_argument('--config_path', default=None)
    args = parser.parse_args()

    # if args.config_path:
    #     config = load(open(args.config_path, "rb"))
    #     if config is None:
    #         print("Please provide a config.")
    #     input_dir = config["common_voice_dir"]
    #     output_dir = config["chopped_wavs"]
    #     max_chops = config["max_chops"]

    # Start a spectrogram generator for each class
    # Each generator will scan a directory for audio files and convert them to images
    # adjust this if you have other languages or any language is missing
    languages = [
        "english",
        "german",
        "french",
        "spanish",
    ]
    
    threads = []
    count = 0
    # create dirs
    for language in languages:
        temp_dir = None
        output_dir = None
        if args.wav_dir:
            temp_dir = os.path.join(args.wav_dir, language)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
        if args.img_dir:
            output_dir = os.path.join(args.img_dir, language)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        function_args = (language, args.source, temp_dir, output_dir,
                        args.audio_length_s, args.augment_nu, 
                        args.feature_type , args.feature_frame_size_ms, args.feature_nu)

        print(function_args)
        
        if args.parallelize_processes:
            threads.append( threading.Thread(target=process_audio_dir, args=function_args,
                                            daemon=True) )
        else:
            process_audio_dir(*function_args)

    if args.parallelize_processes:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
