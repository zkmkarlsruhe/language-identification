import os
import numpy as np
import fnmatch
import scipy.io.wavfile as wav


def pad_with_silence(data, max_len):
    to_add = max(max_len - len(data), 0)
    padded = np.pad(data, (0, to_add), mode='constant', constant_values=0)
    return padded


def recursive_glob(path, pattern):
    for root, dirs, files in os.walk(path):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.abspath(os.path.join(root, basename))
                if os.path.isfile(filename):
                    yield filename


class AudioGenerator(object):
    def __init__(self, source, target_length_s, shuffle=True, run_only_once=True, dtype="float32", minimum_length=0.5):
        """
        A class for generating audio samples of equal length either from directory or file

        :param source: may be a file or directory containing audio files
        :param target_length_s: the length of the desired audio chunks in seconds
        :param shuffle: whether to shuffle the list of the directory content before processing
        :param run_only_once: whether to stop after all chunks have been yielded once
        :param dtpye: type of output
        :param minimum_length: minimum length of the audio chunk in percentage
        """
        self.source = source
        self.shuffle = shuffle
        self.target_length_s = target_length_s
        self.run_only_once = run_only_once
        self.dtype = dtype
        self.minimum_length = minimum_length
        if os.path.isdir(self.source):
            files = []
            files.extend(recursive_glob(self.source, "*.wav"))
            if shuffle:
                np.random.shuffle(files)
        else:
            files = [self.source]
        self.files = files

    def get_generator(self):
        """
        returns a generator that iterates over the source directory or file
        the generator yields audio chunks of desired target length, the sampling frequency and file name
        """
        file_counter = 0
        while True:
            file = self.files[file_counter]
            try:
                # read a file and calculate according parameters
                fs, audio = wav.read(file)
                file_name = file.split('/')[-1]
                target_length = self.target_length_s * fs
                num_segments = int(len(audio) // target_length)

                # for all segments create slices of target length
                for i in range(0, num_segments):
                    slice_start = int(i * target_length)
                    slice_end = int(slice_start + target_length)
                    rest = len(audio) - slice_start
                    # if we have only one segment left and there is at least minimum_length% data pad it with silence
                    if i == num_segments:
                        if rest >= target_length * self.minimum_length:
                            chunk = pad_with_silence(audio[slice_start:], target_length)
                        else:
                            break
                    else:
                        chunk = audio[slice_start:slice_end]
                    chunk = chunk.astype(dtype=self.dtype)
                    yield [chunk, fs, file_name]

            except Exception as e:
                print("AudioGenerator Exception: ", e, file)
                pass

            finally:
                file_counter += 1
                if file_counter >= len(self.files):
                    if self.run_only_once:
                        break
                    if os.path.isdir(self.source) and self.shuffle:
                        np.random.shuffle(self.files)
                    file_counter = 0

    def get_num_files(self):
        return len(self.files)


if __name__ == "__main__":

    a = AudioGenerator("../lid_client/test/", 10, shuffle=True, run_only_once=True)
    gen = a.get_generator()

    i = 0
    for data, fs, fn in gen:
        print(data)
        print(len(data))
        if i > 20:
            break
        else:
            # wav.write(filename=(fn[:-4]+str(i)+".wav"), rate=fs, data=data)
            i += 1
