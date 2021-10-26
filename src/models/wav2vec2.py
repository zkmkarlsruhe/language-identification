"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under Simplified BSD License.
"""


from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Dense, Permute, Input, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50

from src.utils.training_utils import get_feature_layer


def create_model(config):

	audio_length_s = config["audio_length_s"] 
	sample_rate = config["sample_rate"]
	feature_nu = config["feature_nu"]
	feature_type = config["feature_type"]
	input_length = audio_length_s * sample_rate


	# input_length = 246000
	# import tensorflow_hub as hub
	# # For using this pre-trained model for training, pass `trainable=True` in `hub.KerasLayer`
	# feature_extractor = hub.KerasLayer("https://tfhub.dev/vasudevgupta7/wav2vec2/1", trainable=True)

	from tensorflow.keras.models import load_model
	model_path = 'wav2vec2'
	feature_extractor = load_model(model_path)

	inputs = Input((input_length), name='input')



	import tensorflow_hub as hub




	model = Sequential()
	model.add(inputs)
	model.add(feature_extractor)
	model.add(BatchNormalization())
	model.add(Flatten())
	# model.add(GlobalAveragePooling2D())
	# model.add(Dropout(0.5))
	model.add(Dense(len(config["languages"]), activation='softmax'))

	return model
