"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under Simplified BSD License.
"""


from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, LayerNormalization
from tensorflow.keras.layers import Dense, Permute, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50

from src.utils.training_utils import get_feature_layer


def create_model(config):

	audio_length_s = config["audio_length_s"] 
	sample_rate = config["sample_rate"]
	feature_nu = config["feature_nu"]
	feature_type = config["feature_type"]
	input_length = audio_length_s * sample_rate

	inputs = Input((input_length, 1), name='input')
	feature_extractor = get_feature_layer(feature_type, feature_nu, sample_rate)
	res_net = ResNet50(include_top=False, weights=None, input_shape=(None,None,1))

	model = Sequential()
	model.add(inputs)
	model.add(feature_extractor)
	model.add(LayerNormalization())
	model.add(Permute((2, 1, 3)))
	model.add(res_net)
	model.add(GlobalAveragePooling2D())
	model.add(Dropout(0.5))
	model.add(Dense(len(config["languages"]), activation='softmax'))

	return model
