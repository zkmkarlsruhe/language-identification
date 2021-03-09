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

from .utils import pad_with_data, pad_with_noise, pad_with_silence, to_array



def tokenize_audio(data, tokenizer, block_size, leading_pause):
    """
    Extract valid tokens from a chunk of audio data

    :param data: is a list of audio frames encoded as bytearray
    :param block_size: is the length of one audio frame
    :param leading_pause: is the number of audio frames to include if a valid token is found
    """

    # concat all buffers except the most recent ones
    joined_data = b''.join(data[leading_pause:])

    # use auditok for tokenization
    ads = ADSFactory.AudioDataSource(audio_source=BufferAudioSource(joined_data),
                                    block_size=block_size)
    ads.open()
    tokens = tokenizer.tokenize(ads)

    audio_list = []
    for t in tokens:
        # use start and end point from token on the actual data and add the initial section
        start = t[1]
        end = t[2] + leading_pause + 1
        data_slice = data[start:end]
        data_sliced_and_joined = b''.join(data_slice)
        # convert to numpy data
        numpy_data = to_array(data_sliced_and_joined, 2, 1)
        audio_list.append(numpy_data)

    return audio_list


def chop_up_audio( file_name, desired_length_s = 10,
                    min_length = 300, max_silence = 200,
                    sample_width = 2, threshold = 60, padding = "Silence",
                    audio_window_ms = 10, leading_pause = 5):
    
    max_length = desired_length_s / audio_window_ms * 1000
    assert(min_length <= max_length)

    # open wav-file stream
    audio_source = ADSFactory.ads(filename=file_name, sw=sample_width, 
                                max_time=None)
    sample_rate = audio_source.get_sampling_rate()
    block_size = int(sample_rate / 1000 * audio_window_ms)  
    audio_source.set_block_size(block_size)
    nn_input_len = desired_length_s * sample_rate

    # create tokenization objects
    validator = AudioEnergyValidator(sample_width=audio_source.get_sample_width(),
                                    energy_threshold=threshold)
    tokenizer = StreamTokenizer(validator=validator, min_length=min_length,
                                max_length=max_length, 
                                max_continuous_silence=max_silence)

    # start reading from the audio source
    audio_buffers = []
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

    # iterate over the list of valid audio windows
    audio_cuttings = []
    chips = tokenize_audio(audio_buffers, tokenizer, block_size, leading_pause)
    for i, numpy_data in enumerate(chips):

        # extend token to desired length
        if padding == "Silence":
            extended_token = pad_with_silence(numpy_data, nn_input_len)
        elif padding == "Data":
            extended_token = pad_with_data(numpy_data, nn_input_len)
        else:
            extended_token = pad_with_noise(numpy_data, nn_input_len)

        file_name_out = os.path.split(file_name)[-1][:-4] + "_" + str(i)
        data_tuple = (file_name_out, sample_rate, extended_token)
        audio_cuttings.append(data_tuple)
        
    return audio_cuttings


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Remove silence form wav-files')
    # Audio Source
    parser.add_argument('--file_name', type=str, required= True,
                        help='file name to read from')
    parser.add_argument('--sample_width', type=int, default=2, choices=(1, 2, 4),
                        help='number of bytes per sample')
    parser.add_argument('--audio_window_ms', type=int, default=10,
                        help='length of an audio frame in milliseconds')
    # Tokenization
    parser.add_argument('--min_length', type=int, default=300,
                        help='minimum number of valid audio frames')
    parser.add_argument('--max_silence', type=int, default=200,
                        help='maximum number of silent audio frames inside one token')
    parser.add_argument('--energy_threshold', type=float, default=60, 
                        help='amount of energy that determines valid audio frames')
    # Buffering
    parser.add_argument('--leading_pause', type=int, default=5,
                        help='number of audio frames to add in front if an event is detected')
    # Neural Network Preprocessing
    padding_choices = ("Silence", "Data", "Noise")
    parser.add_argument('--padding', type=str, default="Data", choices=padding_choices,
                        help='whether to pad extracted tokens  with silence, data or noise')
    parser.add_argument('--audio_length_s', type=int, default=10,
                        help='desired output length')
    # Other
    parser.add_argument('--output_dir', type=str, default="./",
                        help='directory to store wav files to')

    args = parser.parse_args()

    # sampling parameters
    sample_width = args.sample_width
    audio_window_ms = args.audio_window_ms
    file_name = args.file_name

    # buffer parameters
    threshold = args.energy_threshold
    leading_pause = args.leading_pause

    # tokenization parameters
    min_length = args.min_length
    max_silence = args.max_silence
    # desired length should not be smaller than the max length of a token

    audio_length_s = args.audio_length_s
    padding = args.padding

    # others
    output_dir = args.output_dir

    chunk = chop_up_audio(file_name, audio_length_s, min_length, max_length, max_silence,
                sample_width, threshold, padding, audio_window_ms, leading_pause)
    
    for item in chunk:
        file_path = os.path.join(output_dir, item[0] + ".wav")
        wav.write(file_path, item[1], item[2])
