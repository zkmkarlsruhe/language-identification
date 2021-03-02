"""
:author:
Paul Bethge (bethge@zkm.de)
2020

:License:
This package is published under GNU GPL Version 3.
"""

import scipy.io.wavfile as wav
import argparse
import os
import time

from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer, BufferAudioSource

from data.wav.audio_utils import *
from utils.common_utils import *


class AudioActivityDetector:

    def __init__(self, tokenizer: StreamTokenizer):
        self.tokenizer = tokenizer

    def analyse_audio(self, data, block_size, initial_skip):
        """
        Extract valid tokens from a chunk of audio data

        :param data: is a list of audio frames encoded as bytearray
        :param block_size: is the length of one audio frame
        :param initial_skip: is the number of audio frames to include if a valid token is found
        """

        # concat all buffers except the most recent ones
        joined_data = b''.join(data[initial_skip:])

        # use auditok for tokenization
        ads = ADSFactory.AudioDataSource(audio_source=BufferAudioSource(joined_data), block_size=block_size)
        ads.open()
        tokens = self.tokenizer.tokenize(ads)

        audio_list = []
        for i, t in enumerate(tokens):

            # use start and end point from token on the actual data and add the initial section
            start = t[1]
            end = t[2] + initial_skip + 1
            data_slice = data[start:end]
            data_sliced_and_joined = b''.join(data_slice)
            # convert to numpy data
            numpy_data = to_array(data_sliced_and_joined, 2, 1)
            audio_list.append(numpy_data)

        return audio_list

def analyse_speech(data: list, detector: AudioActivityDetector, **kwargs):
    """
     This function first looks for acoustic activity inside the given data.
    Then each valid audio window is saved as wav file.

    :param data: is a list of audio frames encoded as bytearray
    :param detector: is the object used for acoustic Activity detection
    :param kwargs: \
    sample_rate: is the sampling frequency \
    block_size: is the length of one audio frame \
    initial_skip: is the number of audio frames to include if a valid token is found
    """

    file_name = kwargs.pop("file_name")
    sample_rate = kwargs.pop("sample_rate")
    block_size = kwargs.pop("block_size")
    initial_skip = kwargs.pop("initial_skip")
    padding = kwargs.pop("padding")
    output_dir = kwargs.pop("output_dir")

    # iterate over the list of valid audio windows
    predictions = []
    for numpy_data in detector.analyse_audio(data, block_size, initial_skip):

        # extend token to desired length
        if padding == "Silence":
            extended_token = pad_with_silence(numpy_data, nn_input_len)
        elif padding == "Data":
            extended_token = pad_with_data(numpy_data, nn_input_len)
        else:
            extended_token = pad_with_noise(numpy_data, nn_input_len)

        # save data
        file_name_out = file_name.split("/")[-1][:-4] + "_" + str(time.time()) + ".wav"
        file_path = os.path.join(output_dir, file_name_out)
        wav.write(file_path, sample_rate, extended_token)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Chop up and analyze audio from file')

    # Tokenization
    parser.add_argument('--min_length', type=int, default=300,
                        help='minimum number of valid audio frames')
    parser.add_argument('--max_length', type=int, default=1000,
                        help='maximum number of audio frames for the same token')
    parser.add_argument('--max_silence', type=int, default=200,
                        help='maximum number of silent audio frames inside one token')
    parser.add_argument('--energy_threshold', type=float, default=60, choices=Range(1, 100),
                        help='amount of energy that determines valid audio frames')
    # Buffering
    parser.add_argument('--initial_skip', type=int, default=5, choices=Range(1, 200),
                        help='number of audio frames to add in front if an event is detected')
    # Audio Source
    parser.add_argument('--file_name', type=str,
                        default="../test/wav/paul_deutsch.wav",
                        help='filename to read from')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='sampling frequency when using the microphone')
    parser.add_argument('--max_time', type=int, default=None,
                        help='maximum recording time')
    # parser.add_argument('--sound_device', type=int, default=0,
    #                     help='choose which input device to record from')
    parser.add_argument('--sample_width', type=int, default=2, choices=(1, 2, 4),
                        help='number of bytes per sample')
    parser.add_argument('--audio_window_ms', type=int, default=10,
                        help='length of an audio frame in milliseconds')
    # Neural Network Preprocessing
    padding_choices = ("Silence", "Data", "Noise")
    parser.add_argument('--padding', type=str, default="Data", choices=padding_choices,
                        help='whether to pad extracted tokens  with silence, data or noise')
    parser.add_argument('--nn_input_len_s', type=int, default=10,
                        help='desired output length')
    # Other
    parser.add_argument('--output_dir', type=str, default="recordings/",
                        help='directory to store wav files to')

    args = parser.parse_args()

    # sampling parameters
    sample_width = args.sample_width
    audio_window_ms = args.audio_window_ms
    file_name = args.file_name

    # buffer parameters
    threshold = args.energy_threshold
    initial_skip = args.initial_skip

    # tokenization parameters
    min_length = args.min_length
    max_length = args.max_length
    max_silence = args.max_silence

    # others
    output_dir = args.output_dir

    # open wav-file stream
    audio_source = ADSFactory.ads(filename=file_name, sw=sample_width, 
                                max_time=args.max_time)
    sample_rate = audio_source.get_sampling_rate()
    block_size = int(sample_rate / 1000 * audio_window_ms)  
    audio_source.set_block_size(block_size)

    # Neural Network Preprocessing
    nn_input_len = args.nn_input_len_s * sample_rate
    padding = args.padding

    # prepare keyword arguments for speech analysis function call
    parameters = {"sample_rate": sample_rate, "block_size": block_size,
                  "initial_skip": initial_skip, "padding": padding,
                  "output_dir": output_dir, "file_name": file_name}



    # create tokenization objects
    validator = AudioEnergyValidator(sample_width=audio_source.get_sample_width(),
                                    energy_threshold=threshold)
    tokenizer = StreamTokenizer(validator=validator, min_length=min_length,
                                max_length=max_length, max_continuous_silence=max_silence)
    detector = AudioActivityDetector(tokenizer)

    audio_buffers = []

    # start reading from the audio source
    audio_source.open()
    while True:

        try:
            frame = audio_source.read()
        except OSError as e:
            print(e)
            continue
        if frame is None:
            break
        audio_buffers.append(frame)

    args = (audio_buffers, detector)
    analyse_speech(*args, **parameters)






        
    # while True:

    #     try:
    #         frame = audio_source.read()
    #     except OSError as e:
    #         print(e)
    #         continue

    #     if frame and len(frame):
    #         fifo.shift_in(frame)
    #     else:
    #         audio_source_empty = True

    #     # if we already analyzed a chunk of data
    #     # and there is no more data to fetch
    #     # empty a portion of the fifo for the last run
    #     if not first_run and audio_source_empty:
    #         for i in range(num_buffers - num_iters):
    #             fifo.shift_out()

    #     # after gathering num_iters buffers analyze the window
    #     if i == num_iters or audio_source_empty:
    #         first_run = False

    #         # copy the data such that the thread is working on a different memory location
    #         data = fifo.data().copy()
    #         if len(data):
    #             args = (data, detector, analyzer)
    #             if real_time:
    #                 # start a thread that handles the computation
    #                 thread_0 = threading.Thread(target=analyse_speech, args=args, kwargs=parameters, daemon=True)
    #                 thread_0.start()
    #             else:
    #                 analyse_speech(*args, **parameters)

    #         # reset the counter
    #         i = 0

    #         if audio_source_empty:
    #             break

    #     i += 1