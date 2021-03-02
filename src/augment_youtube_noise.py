import argparse
import os

import numpy as np
import scipy.io.wavfile as wav

from data.audio.generators import AudioGenerator


def augment_noise(args):

    # get params
    source_dir = args.source
    target_length_s = args.target_length_s

    wav_dir = args.wav_dir
    num_files = args.num_files

    num_generators = 4
    mix_length = target_length_s / num_generators

    # create generators
    generators = [AudioGenerator(source=source_dir,
                                 target_length_s=mix_length,
                                 dtype="float32",
                                 shuffle=True, run_only_once=False)
                  for num in range(num_generators)]
    generator_queues = [audioGen.get_generator() for audioGen in generators]

    if wav_dir:
        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)

    for i in range(num_files):
        data, fs, _ = next(generator_queues[0])
        for j in range(1, num_generators):
            data_other, fs_other, _ = next(generator_queues[j])
            # once in a while skip a data chunk (desync)
            if np.random.randint(0, 4) < 1:
                data_other, fs_other, _ = next(generator_queues[j])
            if fs != fs_other:
                print("mixing signals with different sampling frequencies is not implemented yet")
                continue
            data = np.append(data, data_other, 0)

        file_name = "noise" + str(i) + ".wav"
        file_name = os.path.join(wav_dir, file_name)
        data = data.astype("int16")
        wav.write(file_name, fs, data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str,
                        default="/home/paul/test_noise",
                        # required=True,
                        help="directory to search for language folders")
    parser.add_argument('--target_length_s', type=int, default=10,
                        help="length of the audio window to process in seconds")

    parser.add_argument('--wav_dir', type=str,
                        default="/home/paul/test_noise_out",
                        help="directory to save processed wav files")

    parser.add_argument('--num_files', type=int, default=30000,
                        help="number of files to output")
    parser.add_argument('--num_generators', type=int, default=4,
                        help="number of random draws - defines length of random drawn chunk")

    cli_args = parser.parse_args()

    augment_noise(cli_args)
