"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under Simplified BSD License.
"""

import tensorflow.keras.backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Dense, Permute, Input, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50

from src.utils.training_utils import get_feature_layer
from src.models.utils import transformer_classifier


def create_model(config):

	audio_length_s = config["audio_length_s"] 
	sample_rate = config["sample_rate"]
	feature_nu = config["feature_nu"]
	feature_type = config["feature_type"]
	input_length = audio_length_s * sample_rate

	inputs = Input((input_length, 1), name='input')
	feature_extractor = get_feature_layer(feature_type, feature_nu, sample_rate)
	sqz = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')
	trans = transformer_classifier(
			num_layers=8,
			d_model=feature_nu,
			num_heads=8,
			dff=256,
			maximum_position_encoding=2048,
			n_classes=len(config["languages"]))

	model = Sequential()
	model.add(inputs)
	model.add(feature_extractor)
	model.add(sqz)
	model.add(BatchNormalization())
	model.add(trans)

	return model
