from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, Bidirectional
from tensorflow.keras.layers import Reshape, Permute, Lambda, Dot, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB7

import tensorflow.keras.backend as K

from kapre.composed import get_stft_magnitude_layer
# from kapre.utils import Normalization2D


NAME = "ATTENTIONRNN"


def create_model(config):

	#input_shape = config["input_shape"]
	#inputs = Input(shape=input_shape, name='input')
	
	audio_length_s = config["audio_length_s"] 
	sample_rate = config["sample_rate"]
	feature_nu = config["feature_nu"]
	input_length = audio_length_s * sample_rate

	inputs = Input((input_length, 1), name='input')
	# x = Reshape((1, -1))(inputs)
	
	# m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, input_length), 
	#         padding='same', sr=sample_rate, n_mels=feature_nu, fmin=40.0, fmax=sample_rate/2,
	#         power_melgram=1.0, return_decibel_melgram=True, trainable_fb=False,
	#         trainable_kernel=False, name='mel_stft')

	m = get_stft_magnitude_layer(input_shape=(input_length, 1), n_fft = 2048,
								return_decibel=True, name='stft_deb')

	# m = STFT(n_fft=1024, win_length=2018, hop_length=1024,
	#            window_name=None, pad_end=False,
	#            input_data_format='channels_last', output_data_format='channels_last',
	#            input_shape=input_shape)

	m.trainable = False
	x = m(inputs)
	# x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)

	# note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
	# we would rather have it the other way around for LSTMs

	# x = Permute((2, 1, 3))(x)
	# x = Reshape((94,80)) (x) #this is strange - but now we have (batch_size,
	# sequence, vec_dim)

	x = Conv2D(10, (5, 1), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = Conv2D(1, (5, 1), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)

	# x = Reshape((125, 80)) (x)
	# keras.backend.squeeze(x, axis)
	x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)

	x = Bidirectional(LSTM(64, return_sequences=True)
						)(x)  # [b_s, seq_len, vec_dim]
	x = Bidirectional(LSTM(64, return_sequences=True)
						)(x)  # [b_s, seq_len, vec_dim]

	xFirst = Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
	query = Dense(128)(xFirst)

	# dot product attention
	attScores = Dot(axes=[1, 2])([query, x])
	attScores = Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

	# rescale sequence
	attVector = Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

	x = Dense(64, activation='relu')(attVector)
	x = Dense(32)(x)

	output = Dense(len(config["languages"]), activation='softmax', name='output')(x)

	model = Model(inputs=[inputs], outputs=[output])

	return model
