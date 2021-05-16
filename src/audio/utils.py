"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under GNU GPL Version 3.
"""

from array import array
import numpy as np

FORMAT = {1: "b", 2: "h", 4: "i"}


def pad(data, max_len, type):
        if type == "Silence":
            return pad_with_silence(data, max_len)
        elif type == "Data":
            return pad_with_data(data, max_len)
        else:
            return pad_with_noise(data, max_len)


def pad_with_silence(data, max_len):
    to_add = max(max_len - len(data), 0)
    padded = np.pad(data, (0, to_add), mode='constant', constant_values=0)
    return padded


def pad_with_data(data, max_len):
    to_add = max(max_len - len(data), 0)
    padded = np.zeros(shape=(max_len,), dtype="int16")
    if to_add:
        repeat = int(max_len / len(data))
        rest = max_len % len(data)
        for i in range(repeat):
            start = i * len(data)
            end = (i+1) * len(data)
            padded[start:end] = data[:]
        # padded[repeat*len(data):] = data[:rest]
        pad_with_silence(padded, max_len)
        return padded
    return data


def pad_with_noise(data, max_len):
    print("padding with noise not implemented yet... padding with silence")
    return pad_with_silence(data, max_len)


def separate_channels(data, fmt, channels):
    all_channels = array(fmt, data)
    mono_channels = [
        array(fmt, all_channels[ch::channels]) for ch in range(channels)
    ]
    return mono_channels


def to_array(data, sample_width, channels):
    fmt = FORMAT[sample_width]
    if channels == 1:
        return np.array(array(fmt, data))
    return separate_channels(data, fmt, channels)

