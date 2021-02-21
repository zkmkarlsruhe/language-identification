"""
:author:
Paul Bethge (bethge@zkm.de)
2020

:License:
This package is published under GNU GPL Version 3.
"""

import scipy.io.wavfile as wav
import threading
import argparse
import os
import time

from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer

from lid_client.src.simple_fifo import SimpleFIFO
from lid_client.src.audio_activity_detection import *
from lid_client.src.audio_utils import *
from lid_client.src.common_utils import *


def analyse_speech(data: list, detector: AudioActivityDetector, analyzer, **kwargs):
    """
    Detect the language from a list of audio frames.

    This function first looks for acoustic activity inside the given data.
    Then each valid audio window is saved as wav file and send to a
    model which determines the most likely language from that file.

    :param data: is a list of audio frames encoded as bytearray
    :param detector: is the object used for acoustic Activity detection
    :param analyzer: is the object which encapsulates the neural network
    :param kwargs: \
    sample_rate: is the sampling frequency \
    block_size: is the length of one audio frame \
    initial_skip: is the number of audio frames to include if a valid token is found \
    nn_input_len: is the number of samples per valid token. A token may be padded \
    nn_min_likelihood: is the minimum likelihood for a language
    """

    file_name = kwargs.pop("file_name")
    sample_rate = kwargs.pop("sample_rate")
    block_size = kwargs.pop("block_size")
    initial_skip = kwargs.pop("initial_skip")
    nn_input_len = kwargs.pop("nn_input_len")
    nn_min_likelihood = kwargs.pop("nn_min_likelihood")
    padding = kwargs.pop("padding")
    nn_disable = kwargs.pop("nn_disable")
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
        if file_name is None:
            file_name_out = str(time.time()) + ".wav"
        else:
            file_name_out = file_name.split("/")[-1][:-4] + "_" + str(time.time()) + ".wav"
        file_path = os.path.join(output_dir, file_name_out)
        wav.write(file_path, sample_rate, extended_token)

        if not nn_disable:
            response = analyzer.predict_on_audio(extended_token, sample_rate)
            predictions.append(response)

    # determine language
    if not nn_disable and len(predictions):
        predictions_avg = analyzer.calculate_average(predictions)
        language_index = analyzer.determine_language_from_list(predictions_avg, nn_min_likelihood)
        if language_index == -1:
            print("None")
        else:
            print(analyzer.get_languages()[language_index])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Chop up and analyze audio either from file or from mic')

    # Tokenization
    parser.add_argument('--min_length', type=int, default=300,
                        help='minimum number of valid audio frames')
    parser.add_argument('--max_length', type=int, default=1000,
                        help='maximum number of audio frames for the same token')
    parser.add_argument('--max_silence', type=int, default=200,
                        help='maximum number of silent audio frames inside one token')
    parser.add_argument('--energy_threshold', type=float, default=60, choices=Range(1, 100),
                        help='amount of energy that determines valid audio frames')
    # Neural Network
    parser.add_argument('--nn_disable', action='store_true', default=False,
                        help='whether to enable the neural network')
    padding_choices = ("Silence", "Data", "Noise")
    parser.add_argument('--padding', type=str, default="Data", choices=padding_choices,
                        help='whether to pad extracted tokens  with silence, data or noise')
    parser.add_argument('--nn_input_len_s', type=int, default=10,
                        help='length of audio windows to be analyzed by the Neural Network')
    parser.add_argument('--nn_model_path', type=str,
                        default="lid_network/trained_models/lang5_eff_acc92/",
                        help='path to the TF Model to load the Neural Network')
    parser.add_argument('--nn_min_likelihood', type=float, default=0.8, choices=Range(0.5, 1.0),
                        help='minimum confidence of the Neural Network to change language')
    # Buffering
    parser.add_argument('--num_buffers', type=int, default=1000,
                        help='number of audio frames to buffer')
    parser.add_argument('--num_iters', type=int, default=600,
                        help='number of audio frames to gather before analysing, overlap is num_buffers - num_iters')
    parser.add_argument('--initial_skip', type=int, default=5, choices=Range(1, 200),
                        help='number of audio frames to add in front if an event is detected')
    # Audio Source
    parser.add_argument('--file_name', type=str,
                        # default=None,
                        default="lid_client/test/paul_deutsch.wav",
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
    # Other
    parser.add_argument('--output_dir', type=str, default="lid_client/recordings/",
                        help='directory to store wav files to')

    args = parser.parse_args()

    # sampling parameters
    sample_width = args.sample_width
    audio_window_ms = args.audio_window_ms
    file_name = args.file_name
    if file_name:
        # open wav-file stream
        audio_source = ADSFactory.ads(filename=file_name, sw=sample_width, max_time=args.max_time)
        sample_rate = audio_source.get_sampling_rate()
        real_time = False
    else:
        sample_rate = args.sample_rate
        # open mic stream
        audio_source = ADSFactory.ads(sw=sample_width, sr=sample_rate, max_time=args.max_time)
        real_time = True
    block_size = int(sample_rate / 1000 * audio_window_ms)  # number of samples per audio frame
    audio_source.set_block_size(block_size)

    # buffer parameters
    threshold = args.energy_threshold
    initial_skip = args.initial_skip
    num_iters = args.num_iters
    num_buffers = args.num_buffers

    # tokenization parameters
    min_length = args.min_length
    max_length = args.max_length
    max_silence = args.max_silence

    # neural network parameters
    nn_disable = args.nn_disable
    model_path = args.nn_model_path
    nn_input_len = args.nn_input_len_s * sample_rate
    nn_min_likelihood = args.nn_min_likelihood
    padding = args.padding

    # if needed load the neural network and warm-up
    if not nn_disable:
        from lid_client.src.language_analyzer import LanguageAnalyzer
        analyzer = LanguageAnalyzer(model_path)
        print("LID sys: starting warm-up...")
        analyzer.predict_on_audio_file("lid_client/test/paul_deutsch.wav")
        print("LID sys: finished warm-up!")
    else:
        analyzer = None

    # others
    output_dir = args.output_dir
    visualize = False
    plot_fft = False

    # assertions
    assert (num_iters <= num_buffers)

    # prepare keyword arguments for speech analysis function call
    parameters = {"sample_rate": sample_rate, "block_size": block_size,
                  "initial_skip": initial_skip, "nn_input_len": nn_input_len,
                  "nn_min_likelihood": nn_min_likelihood, "padding": padding,
                  "nn_disable": nn_disable, "output_dir": output_dir,
                  "file_name": file_name}

    # create a FIFO storing audio frames
    fifo = SimpleFIFO(max_len=num_buffers)

    # create tokenization objects
    validator = AudioEnergyValidator(sample_width=audio_source.get_sample_width(), energy_threshold=threshold)
    tokenizer = StreamTokenizer(validator=validator, min_length=min_length,
                                max_length=max_length, max_continuous_silence=max_silence)
    detector = AudioActivityDetector(tokenizer)

    i = 0
    last_frame_read = False
    audio_source_empty = False
    first_run = True

    # start reading from the audio source
    audio_source.open()
    while True:

        try:
            frame = audio_source.read()
        except OSError as e:
            print(e)
            continue

        if frame and len(frame):
            fifo.shift_in(frame)
        else:
            audio_source_empty = True

        # if we already analyzed a chunk of data
        # and there is no more data to fetch
        # empty a portion of the fifo for the last run
        if not first_run and audio_source_empty:
            for i in range(num_buffers - num_iters):
                fifo.shift_out()

        # after gathering num_iters buffers analyze the window
        if i == num_iters or audio_source_empty:
            first_run = False

            # copy the data such that the thread is working on a different memory location
            data = fifo.data().copy()
            if len(data):
                args = (data, detector, analyzer)
                if real_time:
                    # start a thread that handles the computation
                    thread_0 = threading.Thread(target=analyse_speech, args=args, kwargs=parameters, daemon=True)
                    thread_0.start()
                else:
                    analyse_speech(*args, **parameters)

            # reset the counter
            i = 0

            if audio_source_empty:
                break

        i += 1
