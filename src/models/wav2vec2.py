"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under Simplified BSD License.
"""



from tensorflow.keras.layers import Permute, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, Bidirectional, Flatten
from tensorflow.keras.layers import LayerNormalization, Lambda, Dot, Softmax
from tensorflow.keras.models import Model


from src.utils.training_utils import get_feature_layer


def create_model(config):

	audio_length_s = config["audio_length_s"] 
	sample_rate = config["sample_rate"]
	input_length = audio_length_s * sample_rate


	# input_length = 246000
	# import tensorflow_hub as hub
	# # For using this pre-trained model for training, pass `trainable=True` in `hub.KerasLayer`
	# feature_extractor = hub.KerasLayer("https://tfhub.dev/vasudevgupta7/wav2vec2/1", trainable=True)

	from tensorflow.keras.models import load_model
	model_path = 'wav2vec2'
	feature_extractor = load_model(model_path)
	feature_extractor.trainable = True

	inputs = Input((input_length), name='input')

	x = feature_extractor(inputs)
	x = BatchNormalization()(x)

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
