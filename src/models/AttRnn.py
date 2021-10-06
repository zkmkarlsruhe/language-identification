"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under Simplified BSD License.
"""

from tensorflow.keras.layers import Permute, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, Bidirectional
from tensorflow.keras.layers import LayerNormalization, Lambda, Dot, Softmax
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K

from src.utils.training_utils import get_feature_layer


def create_model(config):
	""" 
	This code is heavily borrowed from <https://github.com/douglas125/SpeechCmdRecognition>.
	"""

	audio_length_s = config["audio_length_s"] 
	sample_rate = config["sample_rate"]
	feature_nu = config["feature_nu"]
	feature_type = config["feature_type"]
	input_length = audio_length_s * sample_rate

	feature_extractor = get_feature_layer(feature_type, feature_nu, sample_rate)

	input_shape = (input_length, 1)
	if feature_type == "yamnet": # the desired output are the features
		input_shape = (None, 1024, 1)
	elif  feature_type == "trill":
		input_shape = (None, 512, 1)

	inputs = Input(input_shape, name='input')

	if not feature_type == "yamnet" and not feature_type == "trill":
		x = feature_extractor(inputs)

	if feature_type == "yamnet": # the desired output are the features
		x = inputs
	elif feature_type == "trill": # the desired output are the features
		x = inputs

	x = BatchNormalization()(x)
	x = Conv2D(10, (5, 1), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = Conv2D(1, (5, 1), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)

	x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)

	x = Bidirectional(LSTM(64, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]
	x = Bidirectional(LSTM(64, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]

	xFirst = Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
	query = Dense(128)(xFirst)

	# dot product attention
	attScores = Dot(axes=[1, 2])([query, x])
	attScores = Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

	# rescale sequence
	attVector = Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

	x = Dense(64, activation='relu')(attVector)	
	x = Dropout(0.25)(x)
	x = Dense(32)(x)
	x = Dropout(0.25)(x)


	output = Dense(len(config["languages"]), activation='softmax', name='output')(x)

	model = Model(inputs=[inputs], outputs=[output])

	return model
