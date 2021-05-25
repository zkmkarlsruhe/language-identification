"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under GNU GPL Version 3.
"""

import os
import random

import numpy as np
import python_speech_features as feat
import scipy.io.wavfile as wav
from imageio import imwrite, imread
from subprocess import Popen, PIPE, STDOUT


def normalize(signal):
    """
    normalize a float signal to have a maximum absolute value of 1.0
    """
    maximum = max(abs(signal.max()), abs(signal.min()))
    if maximum == 0.0:
        return signal
    return signal / float(maximum)


def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def mfcc(signal, fs, win_step_s, num_features):
    data = feat.mfcc(signal, fs, winstep=win_step_s, numcep=num_features)
    return np.moveaxis(data, [0, 1], [1, 0])


def logfbank(signal, fs, win_step_s, num_features):
    data = feat.logfbank(signal, fs, winstep=win_step_s, nfilt=num_features)
    return np.moveaxis(data, [0, 1], [1, 0])


def logfbankenergy(signal, fs, win_step_s, num_features):
    data, energy = feat.fbank(signal, fs, winstep=win_step_s, nfilt=num_features - 1)
    energy = np.expand_dims(energy, axis=1)
    data = np.concatenate((data, energy), axis=1)
    data = np.log(data + 0.000001)
    return np.moveaxis(data, [0, 1], [1, 0])


def sox_spectrogram_from_file(file_name, fs_desired, len_segment_ms, num_features):
    """
    compute the spectrogram from file using sox.

    :param fs_desired: is the frequency at which the file is resampled

    """
    pixel_per_second = 1.0 / len_segment_ms
    file_out = "tmp_{}.png".format(random.randint(0, 100000))
    command = "sox -V0 '{}' -n remix 1 rate {} spectrogram -y {} -X {} -m -r -o {}".format(
        file_name, fs_desired, num_features,
        pixel_per_second,
        file_out)
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    output, errors = p.communicate()
    if errors:
        print(errors)
    image = imread(file_out)
    os.remove(file_out)
    return np.array(image, dtype="float")


def sox_spectrogram_from_array(signal, fs, fs_desired, len_segment_ms, num_features):
    assert num_features >= 64, "sox spectrogram requires at least 64 frequency bins"
    file_name = "tmp_{}.wav".format(random.randint(0, 100000))
    wav.write(file_name, fs, signal)
    array = sox_spectrogram_from_file(file_name, fs_desired, len_segment_ms, num_features)
    os.remove(file_name)
    return array


def signal_to_features(signal, fs, len_segment_ms=20, num_features=128,
                       zero_center=False, remap=(0, 1), audio_feature="spectrogram"):

    """
    compute speech features for a given signal

    :param signal: array containing the signal
    :param fs: sampling frequency
    :param len_segment_ms: length of the audio frame in ms
    :param num_features: number of speech features to compute. For spectrogram this should be above 64
    :param zero_center: whether to subtract the mean of the output image from every pixel
    :param remap: determine the range of the output image. Spectrogram will have [0, 255] by default,
     others might have negative values as well.
    :param audio_feature: could be "mfcc", "logfbank", logfbankenergy" (logfbank + signal energy), "spectrogram"
    """

    win_step_s = len_segment_ms / 1000.0

    if audio_feature == "mfcc":
        data = mfcc(signal, fs, win_step_s, num_features)

    elif audio_feature == "logfbank":
        data = logfbank(signal, fs, win_step_s, num_features)

    elif audio_feature == "logfbankenergy":
        data = logfbankenergy(signal, fs, win_step_s, num_features)

    elif audio_feature == "spectrogram":
        data = sox_spectrogram_from_array(signal, fs, 10000, win_step_s, num_features)

    else:
        print("Please specify an audio feature")
        return None

    if zero_center:
        data -= np.mean(data)

    if remap:
        data = map_range(data, data.min(), data.max(), remap[0], remap[1])

    return data.astype(dtype="float32")


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_file = "../../../language-identification-system/lid_client/test/paul_deutsch.wav"
    fs, audio = wav.read(test_file)

    mfcc_array = signal_to_features(audio, fs, 20, 13, audio_feature="mfcc")
    fbank_array = signal_to_features(audio, fs, 20, 40, audio_feature="logfbank")
    fbankenenergy_array = signal_to_features(audio, fs, 20, 40, audio_feature="logfbankenergy")
    spec_array = signal_to_features(audio, fs, 20, 128, audio_feature="spectrogram", remap=None, zero_center=True)
    # spec_array = signal_to_features(audio, fs, 20, 128, audio_feature="spectrogram")

    mfcc_array = (mfcc_array*255).astype("uint8")
    fbank_array = (fbank_array*255).astype("uint8")
    fbankenenergy_array = (fbankenenergy_array*255).astype("uint8")
    spec_array = (spec_array*255).astype("uint8")

    imwrite("mfcc.png", mfcc_array)
    imwrite("fbank.png", fbank_array)
    imwrite("fbankenenergy.png", fbankenenergy_array)
    imwrite("log_spec_array.png", spec_array)

    time = np.arange(len(audio))

    plt.figure()

    plt.subplot(611)
    plt.plot(time, audio)

    plt.subplot(612)
    plt.imshow(mfcc_array)

    plt.subplot(613)
    plt.imshow(fbank_array)

    plt.subplot(614)
    plt.imshow(fbankenenergy_array)

    plt.subplot(615)
    plt.imshow(spec_array)

    plt.subplot(616)
    plt.specgram(audio, Fs=fs)

    plt.tight_layout()

    plt.show()
