
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3

NAME = "INC"


def create_model(config):

    input_shape = config["input_shape"]
    inception_model = InceptionV3(input_shape=input_shape,
                                  include_top=False,
                                  weights=None,
                                  )

    x = GlobalAveragePooling2D()(inception_model.output)
    x = Dropout(0.5)(x)
    predictions = Dense(len(config["languages"]), activation='softmax')(x)

    return Model(inputs=inception_model.input, outputs=predictions)
