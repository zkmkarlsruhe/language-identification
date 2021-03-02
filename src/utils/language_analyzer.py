import numpy as np
import tensorflow as tf
from imageio import imread
from yaml import load

# from data.generators import SpectrogramGenerator

from data import audio_features
from data.generators import AudioGenerator
import lid_network.models as models


class LanguageAnalyzer:

    def __init__(self, model_path, config_path=None):

        if config_path is None:
            config_path = model_path + "config.yaml"

        # read config
        config = load(open(config_path, "rb"))
        if config is None:
            print("Please provide a config.")

        # save hyperparameters
        self._audio_feature = config["audio_feature"]
        self._audio_length_s = config["audio_length_s"]
        self._input_height = config["input_shape"][0]
        self._input_width = config["input_shape"][1]
        self._len_segment_ms = config["len_segment_ms"]
        self._languages = config["languages"]
        self._model_name = config["model"]
        self._remap = config["remap_img"]
        self._zero_center = config["zero_center_img"]

        # create the model
        model_class = getattr(models, self._model_name)
        self._model = model_class.create_model(config)
        self._model.load_weights(model_path+"variables/variables")

    def _predict(self, data):
        predictions = self._model.predict(data)
        lang_index = np.argmax(predictions[0])
        print("Predicted: ", self._languages[lang_index], " with ", predictions[0][lang_index] * 100, "% certainty")
        return predictions[0]

    def predict_on_image(self, img_array):
        img_array = tf.keras.preprocessing.image.img_to_array(img_array)
        img_array = tf.expand_dims(img_array, 0)
        return self._predict(img_array)

    def predict_on_image_file(self, img_path, normalize=True):
        # img = load_img(img_path, target_size=(self._input_height, self._input_width, 1), color_mode="grayscale")
        img = imread(img_path, mode="L")
        if normalize:
            img /= 255.0
        return self.predict_on_image(img)

    def predict_on_audio(self, audio_chunk, fs):

        # old model (lang5_eff_acc92)
        img = audio_features.signal_to_features(signal=audio_features.normalize(audio_chunk), fs=fs,
                                                len_segment_ms=self._len_segment_ms, num_features=self._input_height,
                                                audio_feature=self._audio_feature, remap=self._remap,
                                                zero_center=self._zero_center)
        return self.predict_on_image(img)

        # new model (lang5_eff_acc94)
        # img = audio_features.signal_to_features(signal=audio_features.normalize(audio_chunk), fs=fs,
        #                                         len_segment_ms=self._len_segment_ms, num_features=self._input_height,
        #                                         audio_feature=self._audio_feature, normalize=(0, 1), zero_center=False)
        # return self.predict_on_image(img, normalize=False)

    def predict_on_audio_file(self, audio_path):
        generator = AudioGenerator(source=audio_path, target_length_s=self._audio_length_s,
                                   dtype="float32", run_only_once=True).get_generator()
        return [self.predict_on_audio(chunk, fs) for chunk, fs, name in generator]

    def get_languages(self):
        return self._languages

    @staticmethod
    def calculate_average(predictions):
        mean = np.mean(predictions, axis=0)
        return mean

    @staticmethod
    def determine_language_from_dict(predictions, min_likelihood):
        max_likelihood = max(predictions.values())
        most_likely_lang = max(predictions, key=predictions.get)
        print("max likelihood: ", max_likelihood, " ", most_likely_lang)
        if max_likelihood >= min_likelihood:
            return most_likely_lang
        else:
            return "English"

    @staticmethod
    def determine_language_from_list(predictions, min_likelihood):
        index = np.argmax(predictions)
        max_likelihood = predictions[index]
        if max_likelihood >= min_likelihood:
            return index
        else:
            return -1
