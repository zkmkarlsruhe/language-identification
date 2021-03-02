import argparse
import pydub
import os
import threading
import scipy.io.wavfile as wav
import numpy as np
from data.wav.chop_up import chop_up_audio

def sentence_is_too_short(sentence_len, language):
    if language == "mandarin":
        return sentence_len < 5
    else:
        return sentence_len < 10


def traverse_csv(language, input_dir, output_dir, total_chops, allowed_downvotes,
                remove_raw):

    lang = language["lang"]
    lang_abb = language["dir"]

    input_sub_dir = os.path.join(input_dir, lang_abb)
    input_sub_dir_clips = os.path.join(input_sub_dir, "clips")


    splits = ["train", "dev", "test"]

    for split_index, split in enumerate(splits):

        output_dir_wav = os.path.join(output_dir, "wav", split, lang)
        output_dir_raw = os.path.join(output_dir, "raw", split, lang)

        # create subdirectories in the output directory
        if not os.path.exists(output_dir_wav):
            os.makedirs(output_dir_wav)
        if not os.path.exists(output_dir_raw):
            os.makedirs(output_dir_raw)

        input_clips_file = os.path.join(input_sub_dir, split + ".tsv")

        # keep track of files handled
        processed_files = 0
        produced_files = 0
        to_produce = total_chops[split_index]
        done = False

        # open mozillas' dataset file
        with open(input_clips_file) as f:

            try:
                # skip the first line
                line = f.readline()

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
                        audio = audio.set_frame_rate(16000)
                        audio = audio.set_channels(1)
                        audio = audio.set_sample_width(2)
                        audio.export(wav_path_raw, format="wav")
                        processed_files += 1

                        # chop up the samples and write to file
                        rand_int = np.random.randint(low=0, high=2)
                        padding_choice = ["Data", "Silence"][rand_int]
                        chips = chop_up_audio (wav_path_raw, padding=padding_choice)
                        for chip in chips:
                            wav_path = os.path.join(output_dir_wav, chip[0] + ".wav")
                            wav.write(wav_path, chip[1], chip[2])
                            produced_files += 1
                            # remove the intermediate file
                            if remove_raw and os.path.exists(wav_path_raw):
                                os.remove(wav_path_raw)
                            # check if we are done yet
                            if to_produce != -1 and produced_files >= to_produce:
                                done = True
                                break

                        if done:
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
    parser.add_argument("--input_dir", type=str, 
    # required=True,
                        default="../data",
                        help="directory containing all languages")
    parser.add_argument("--output_dir", type=str, 
    # required=True,
                        default="../res",
                        help="directory to receive converted clips of all languages")
    parser.add_argument("--total_chops", type=int, nargs=3, default=[-1, -1, -1],
                        help="amount of wav chops to be produced per split")
    parser.add_argument("--allowed_downvotes", type=int, default=0,
                        help="amount of downvotes allowed")
    parser.add_argument("--run_as_thread", type=bool, default=True,
                        help="whether to use multiprocessing")
    parser.add_argument("--remove_raw", type=bool, default=True,
                        help="whether to remove intermediate file")
    parser.add_argument("--use_random_padding", type=bool, default=True,
                        help="whether to use randomly silence or data padding")
    args = parser.parse_args()

    languages = [
        {"lang": "english", "dir": "en"},
        {"lang": "german", "dir": "de"},
        {"lang": "french", "dir": "fr"},
        {"lang": "spanish", "dir": "sp"},
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
    count = 0
    for language in languages:
        # clips_per_language = args.total_chops
        # if language["lang"] == "unknown":
        #     clips_per_language = number_unknown
        function_args = (language, args.input_dir, args.output_dir,
                        args.total_chops, args.allowed_downvotes,
                        args.remove_raw)
        if args.run_as_thread:
            threads.append(threading.Thread(target=traverse_csv, args=function_args,
                                            daemon=True) )
            threads[count].start()
        else:
            traverse_csv(*function_args)
        count += 1

    if args.run_as_thread:
        for t in threads:
            t.join()
