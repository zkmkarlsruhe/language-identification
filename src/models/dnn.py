from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

NAME = "DNN"


def create_model(config):

    input_shape = config["input_shape"]
    model = Sequential()

    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(len(config["languages"]), activation="softmax"))

    return model
