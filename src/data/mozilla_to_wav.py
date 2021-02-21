import argparse
import pydub
import os


def sentence_is_too_short(sentence_len, language):
    if language == "chinese":
        return sentence_len < 5
    else:
        return sentence_len < 10


def traverse_csv(language, input_dir, output_dir, lid_client_path, number, allowed_downvotes):

    lang = language["lang"]
    lang_abb = language["dir"]

    input_sub_dir = os.path.join(input_dir, lang_abb)
    input_sub_dir_clips = os.path.join(input_sub_dir, "clips")
    input_clips_file = os.path.join(input_sub_dir, "validated.tsv")

    output_dir_raw = os.path.join(output_dir, "raw", lang)
    output_dir_pro = os.path.join(output_dir, "pro", lang)
    output_clips_file = os.path.join(output_dir_raw, "clips.csv")

    # create subdirectories in the output directory
    if not os.path.exists(output_dir_raw):
        os.makedirs(output_dir_raw)
    if not os.path.exists(output_dir_pro):
        os.makedirs(output_dir_pro)

    # open the csv file to write to
    out = open(output_clips_file, "w+")

    # open mozillas' dataset file
    with open(input_clips_file) as f:

        # for a certain number extract the file
        i = 0

        try:
            #  skip the first line
            line = f.readline()

            while True:

                # get a line
                line = f.readline().split('\t')

                # if the line is not empty
                if line[0] != "":

                    # check if the sample contains more than X symbols and has not been down voted
                    sentence = line[2]
                    too_short = sentence_is_too_short(len(sentence), language["lang"])
                    messy = int(line[4]) > allowed_downvotes
                    if (too_short or messy) and language is not "unknown":
                        continue

                    # get mp3 filename
                    mp3_filename = line[1]
                    wav_filename = mp3_filename[:-4] + ".wav"
                    mp3_path = os.path.join(input_sub_dir_clips, mp3_filename)
                    wav_path = os.path.join(output_dir_raw, wav_filename)


                    print("\n Processing: ", mp3_path)

                    # convert mp3 tp wav
                    pydub.AudioSegment.from_mp3(mp3_path)\
                        .set_frame_rate(16000)\
                        .set_channels(1)\
                        .export(wav_path, format="wav")

                    # process file through lid_client
                    command = "python " + lid_client_path + \
                        " --file_name " + wav_path + \
                        " --output_dir " + output_dir_pro + \
                        " --num_iters 600" + " --padding Data" + \
                        " --min_length 300" + " " +\
                        " --max_silence 200" " --nn_disable"
                    os.system(command)

                    # save filename to csv
                    line_to_write = wav_filename + '\n'
                    out.write(line_to_write)

                    if number != -1 and i >= number:
                        break
                    i = i+1

                    if i % 1000 == 0:
                        print("Processed %d files", i)

                else:
                    print("Nothing left!")
                    break

        except EOFError as e:
            print("End of file!")

    out.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str,
                        help="directory containing all languages")
    parser.add_argument("--output_dir", type=str,
                        help="directory to receive raw and processed clips of all languages")
    parser.add_argument("--lid_client_path", type=str,
                        default="lid.py",
                        help="path to the python script to process the data")
    parser.add_argument("--number", type=int, default=40000,
                        help="amount of files to be processed (!= amount of processed files)")
    parser.add_argument("--allowed_downvotes", type=int, default=0,
                        help="amount of downvotes allowed")
    args = parser.parse_args()

    languages = [
        # {"lang": "english", "dir": "en"},
        # {"lang": "german", "dir": "de"},
        # {"lang": "french", "dir": "fr"},
        # {"lang": "spanish", "dir": "es"},
        # {"lang": "mandarin", "dir": "zh-CN"},
        # {"lang": "russian", "dir": "ru"},
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
    for language in languages:
        if language["lang"] == "unknown":
            unknown += 1
    if unknown > 0:
        number_unknown = args.number / unknown

    for language in languages:
        clips_per_language = args.number
        if language["lang"] == "unknown":
            clips_per_language = number_unknown
        traverse_csv(language, args.input_dir, args.output_dir, args.lid_client_path,
                     clips_per_language, args.allowed_downvotes)

