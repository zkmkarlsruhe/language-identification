import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import GlobalAveragePooling2D, Convolution1D
from tensorflow.keras.layers import Dense, Permute, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import InceptionV3

NAME = "INCV3BLSTM"


def create_model(config):

    input_shape = config["input_shape"]
    inception_model = InceptionV3(input_shape=input_shape,
                                  include_top=False,
                                  # weights='imagenet',
                                  weights=None,
                                  )

    x = inception_model.output

    # (bs, y, x, c) --> (bs, x, y, c)
    x = Permute((2, 1, 3))(x)

    # (bs, x, y, c) --> (bs, x, y * c)
    _x, _y, _c = [int(s) for s in x._shape[1:]]
    x = Reshape((_x, _y * _c))(x)

    x = Bidirectional(LSTM(512, return_sequences=False), merge_mode="concat")(x)

    predictions = Dense(len(config["languages"]), activation='softmax')(x)

    return Model(inputs=inception_model.input, outputs=predictions)
