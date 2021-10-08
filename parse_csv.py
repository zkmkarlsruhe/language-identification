import pandas as pd
import numpy as np
import math

import os
import pydub
import argparse
from threading import Thread
import numpy as np
import scipy.io.wavfile as wav

import shutil
from yaml import load


def sentence_is_too_short(sentence, language):
    if language == "mandarin":
        return len(sentence) < 3
    else:
        return len(sentence) < 6


def traverse_csv(language, input_dir, output_dir, max_chops, 
                desired_audio_length_s, sample_rate, sample_width,
                allowed_downvotes, remove_raw, min_length_s, max_silence_s,
                energy_threshold, use_vad=True):
    """
    traverses the language specific file, extract and save important samples.
    """
    
    lang = language["lang"]
    lang_abb = language["dir"]

    input_sub_dir = os.path.join(input_dir, lang_abb)
    input_sub_dir_clips = os.path.join(input_sub_dir, "clips")

    splits = ["train", "dev", "test"]
    splits = ["train"]

    # todo: fix path
    # if use_vad:
    #     model_path = "src/audio/silero_vad/model.jit"
    #     vad = VADTokenizer(model_path, min_length_s, desired_audio_length_s, max_silence_s)

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

        input_clips_file  = os.path.join(input_sub_dir, split + ".tsv")
        output_clips_file = os.path.join(output_dir_wav, "clips.tsv")

        column_names = ["path", "age", "gender", "accent", "locale"]
        output_df = pd.DataFrame(columns = column_names)


        # open mozillas' dataset file
        df = pd.read_csv(input_clips_file, sep='\t')

        # sort out unwanted entries
        ### too messy
        df = df[df.down_votes <= allowed_downvotes]
        ### too short
        vec_sentence_is_too_short = np.vectorize(sentence_is_too_short, excluded="language", otypes=[bool])
        df = df[~vec_sentence_is_too_short(df["sentence"], language["lang"])]

        # split data frame by genders
        def is_nan(x):
            return str(x) == "nan"
        vec_is_nan = np.vectorize(is_nan, otypes=[bool])
        unknowns = df[vec_is_nan(df["gender"])]
        females = df[df.gender == 'female']
        males = df[df.gender == 'male']

        # shuffle gender data frames
        males = males.sample(frac = 1)
        females = females.sample(frac = 1)
        unknowns = unknowns.sample(frac = 1)

        # binary gender definition
        FEMALE = 0
        MALE = 1
        UNKNOWN = 2

        # state machine 
        state = 1

        # keep track of how many females and males are already collected
        gender_counter = [0,0,0]
        
        # pools for gender specific data
        males = males.values.tolist()
        females = females.values.tolist()
        unknowns = unknowns.values.tolist()

        genders = [males, females, unknowns]

        # iterate over samples until maximum count is reached or no data
        while not done:

            # sample from gender frames
            # State 1: given enough data we sample from both genders equally
            # State 2: after one is depleated sample from unknown
            # State 3: after unknown is depleated sample from whatever gender is left

            # state signals: do we have data for the genders?
            no_males = not len(genders[MALE])
            no_females = not len(genders[FEMALE])
            no_unknowns = not len(genders[UNKNOWN])

            # break statements
            if sum(gender_counter) >= to_produce:           # done
                break
            elif state == 3 and no_males and no_females:    # nothing left
                break

            # state transitions
            elif state == 1 and (no_males or no_females): # a gender is depleated
                if no_unknowns:
                    state = 3
                else:
                    state = 2
            
            elif state == 2 and no_unknowns: 
                state = 3

            # state action
            # which gender should be picked next? Female (0), male (1), unknown (-1)
            if state == 1: # use the data frame that is underrepresented
                    gender_selected = FEMALE if gender_counter[FEMALE] <= gender_counter[MALE] else MALE
            elif state == 2: # use unknown distribution
                    gender_selected = UNKNOWN
            elif state == 3: # use biased distribution
                    gender_selected = MALE if no_females else FEMALE
            else:
                print("Err unknown state")

            # select a gender specific list and pop the first item
            line = genders[gender_selected].pop()

            # get meta data
            mp3_filename    = line[1]
            age             = line[5]
            gender          = line[6]
            accent          = line[7]
            locale          = line[8]

            # get full mp3 path and wav output filename for conversion
            mp3_path = os.path.join(input_sub_dir_clips, mp3_filename)
            wav_path_raw = os.path.join(output_dir_raw,
                                        mp3_filename[:-4] + ".wav")

            # convert mp3 to wav
            # audio = pydub.AudioSegment.from_mp3(mp3_path)
            # audio = pydub.effects.normalize(audio)
            # audio = audio.set_frame_rate(sample_rate)
            # audio = audio.set_channels(1)
            # audio = audio.set_sample_width(sample_width)
            # audio.export(wav_path_raw, format="wav")
            # processed_files += 1

            # # chop up the samples and write to file
            # rand_int = np.random.randint(low=0, high=2)
            # padding_choice = ["Data", "Silence"][rand_int]

            # if use_vad:
            #     chips = vad.chop_from_file(wav_path_raw, padding=padding_choice)
            # else:
            #     chips = chop_up_audio (wav_path_raw, padding=padding_choice,
            #                     desired_length_s=desired_audio_length_s,
            #                     min_length_s=min_length_s, max_silence_s=max_silence_s,
            #                     threshold=energy_threshold)

            # for chip_name, chip_fs, chip_data in chips:
            #     if chip_data.dtype == "float32":
            #         chip_data = chip_data * 32768
            #         chip_data = chip_data.astype("int16")
            #     wav_path = os.path.join(output_dir_wav, chip_name + ".wav")
            #     wav.write(wav_path, chip_fs, chip_data)
                

            #     # remove the intermediate file
            #     if remove_raw and os.path.exists(wav_path_raw):
            #         os.remove(wav_path_raw)
            #     # check if we are done yet
            #     if sum(gender_counter) >= to_produce:
            #         break
 
            output_df.append(["chip_name" + ".wav", age, gender, accent, locale], ignore_index=True)
            gender_counter[gender_selected] += 1
            # TODO write to csv
            
        produced_files = sum(gender_counter)
        print("Processed %d mp3 files for %s-%s" % (processed_files, lang, split))
        print("Produced  %d wav files for %s-%s" % (produced_files, lang, split))
        print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default=None,
                        help="path to the config yaml file. When given, arguments will be ignored")
    parser.add_argument("--cv_input_dir", type=str, default="common_voice",
                        help="directory containing all languages")
    parser.add_argument("--cv_output_dir", type=str, default="common_voice_processed",
                        help="directory to receive converted clips of all languages")
    # Data 
    parser.add_argument("--max_chops", type=int, nargs=3, default=[100, 100, 100],
                        help="amount of maximum wav chops to be produced per split.")
    parser.add_argument("--allowed_downvotes", type=int, default=0,
                        help="amount of downvotes allowed")
    # Audio file properties
    parser.add_argument("--audio_length_s", type=int, default=5,
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
    parser.add_argument("--use_vad", type=bool, default=True,
                        help="whether to use Silero VAD or Auditok for chopping up")
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
            args.cv_input_dir      = config["cv_input_dir"]
            args.cv_output_dir     = config["cv_output_dir"]
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
            args.use_vad           = config["use_vad"]
            language_table         = config["language_table"]

            # copy config to output dir
            if not os.path.exists(args.cv_output_dir):
                os.makedirs(args.cv_output_dir)
            shutil.copy(args.config_path, args.cv_output_dir)

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
    unknown = 0
    for language in language_table:
        if language["lang"] == "unknown":
            unknown += 1

    threads = []
    for language in language_table:

        max_chops = args.max_chops
        if language["lang"] == "unknown":
            max_chops /= unknown

        # prepare arguments
        function_args = (language, args.cv_input_dir, args.cv_output_dir, args.max_chops, 
                        args.audio_length_s, args.sample_rate, args.sample_width, 
                        args.allowed_downvotes, args.remove_raw, args.min_length_s,
                        args.max_silence_s, args.energy_threshold, args.use_vad)

        # process current language for all splits
        if args.parallelize:
            threads.append(Thread(target=traverse_csv, args=function_args,daemon=True))
        else:
            traverse_csv(*function_args)

    # wait for threads to end
    if args.parallelize:
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    if args.remove_raw:
        shutil.rmtree(os.path.join(args.cv_output_dir, "raw"))
