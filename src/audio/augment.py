"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under GNU GPL Version 3.
"""

import nlpaug.flow as flow
import nlpaug.augmenter.audio as naa


class AudioAugmenter(object):
    def __init__(self, fs):
        self._fs = fs
        shift = naa.ShiftAug(sampling_rate=fs, direction='random', duration=0.2)
        crop = naa.CropAug(sampling_rate=fs, zone=(0.2, 0.8), coverage=0.02)
        vltp = naa.VtlpAug(sampling_rate=fs, zone=(0.2, 0.8), coverage=0.8, 
                            fhi=4800, factor=(0.9, 1.1))
        noise = naa.NoiseAug(zone=(0.0, 1.0), coverage=1, color='white')
        # speed = naa.SpeedAug(zone=(0.0, 1.0), coverage=0.1, factor=(0.9, 1.1))
        # pitch = naa.PitchAug(sampling_rate=16000, zone=(0, 1), coverage=0.3, factor=(0, 2.1))
        self._aug_flow = flow.Sequential([
            shift,
            crop,
            vltp,
            # speed,
            # pitch,
            noise,
        ])

    def augment_audio_array(self, signal, fs):
        assert fs == self._fs
        data = signal.astype(dtype="float32")
        augmented_data = self._aug_flow.augment(data)
        return augmented_data.astype(dtype="float32")


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import scipy.io.wavfile as wav

    file_name = ""
    fs, arr = wav.read(filename=file_name)
    a = AudioAugmenter(fs)

    runs = 4
    plt.figure()
    plt.subplot(runs+1, 1, 1)
    plt.plot(arr)
    for i in range(runs):
        plt.subplot(runs+1, 1, i+2)
        data = a.augment_audio_array(arr, fs)
        plt.plot(data)
    plt.tight_layout()
    plt.show()
