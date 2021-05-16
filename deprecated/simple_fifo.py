"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under GNU GPL Version 3.
"""

class SimpleFIFO:

    """
    A Class for temporarily storing lists of data with shift functionality (FIFO)
    """

    def __init__(self, data=[], max_len: int = 10):
        """
        :param data: is a one-dimensional iterable - one item in the buffer
        :param max_len: defines how many items to keep in the buffer
        """
        assert(max_len > 0)
        self._data = []
        if len(data):
            self._data.append(data)
            self._len = 1
        else:
            self._len = 0
        self._max_len = max_len

    def shift_in(self, new_data):
        """
        Shift in new data and shift out old data if max_len is reached

        :param new_data: is a chunk of values to store
        :return: Empty list or the oldest entry popping out
        """
        ret = []
        if len(new_data):
            if self._len < self._max_len:
                self._len += 1
            else:
                ret = self._data.pop(0)
            self._data.append(new_data)
        return ret

    def shift_out(self):
        """
        Shift out data if not empty
        :return: Empty list or the oldest entry popping out
        """
        ret = []
        if self._len > 0:
            self._len -= 1
            ret = self._data.pop(0)
        return ret

    def data(self):
        """
        :return: access to internal data
        """
        return self._data

    def __str__(self):
        return str(self._data)


def main():
    import numpy as np

    data = np.random.rand(2)
    sf = SimpleFIFO(data, 6)

    for i in np.arange(0, 10, 1):
        data = np.random.rand(2)
        sf.shift_in(data)
        print(sf)


if __name__ == "__main__":
    main()

