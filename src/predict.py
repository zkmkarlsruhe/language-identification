import argparse
from yaml import load
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model


def predict(config_path, model_path, img_path):

    # Config
    config = load(open(config_path, "rb"))
    if config is None:
        print("Please provide a config.")

    img_height = config["input_shape"][0]
    img_width = config["input_shape"][1]
    color_mode = config["color_mode"]

    # load image
    img = load_img(img_path, target_size=(img_height, img_width), color_mode=color_mode)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array /= 255.0

    # show image
    plt.imshow(img_array)
    plt.show()

    # convert image to batch
    img_array = tf.expand_dims(img_array, 0)

    # load model and infer
    model = load_model(filepath=model_path)
    predictions = model.predict(img_array)

    # print output
    languages = ['english', 'french', 'german', 'russian', 'spanish']
    print("Languages: ", languages)
    print("ANN output:", predictions[0])
    print("Predicted: ", languages[np.argmax(predictions[0])])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='trained_models/lang5_eff_acc92')
    parser.add_argument('--input', default='test/mandarin.png')
    parser.add_argument('--config', default="config.yaml")
    cli_args = parser.parse_args()

    predict(cli_args.config, cli_args.model, cli_args.input)
