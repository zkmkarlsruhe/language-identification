import tensorflow as tf
import tensorflow_hub as hub

# yamnet_url = 'https://tfhub.dev/google/yamnet/1'
# layer = hub.KerasLayer(yamnet_url, 
#     trainable=False
#     )
# signal =  tf.keras.layers.Input(shape=(), name='input', dtype=tf.float32)
# outputs = layer(signal)
# model = tf.keras.models.Model(inputs=signal, outputs=outputs)
# model.compile()
# model.summary()

# model.save('yamnet')

from tensorflow import keras
# layer = keras.models.load_model('yamnet')
# print
# signal =  tf.keras.layers.Input(shape=(), name='input', dtype=tf.float32)
# _, features, _ = layer(signal)
# model = tf.keras.models.Model(inputs=signal, outputs=features)
# model.compile()
# model.summary()
# model.save('my_yamnet')


model = keras.models.load_model('my_yamnet')
model.compile()
model.summary()

@tf.function(input_signature=[tf.TensorSpec([None, None], dtype=tf.float32)])
def model_train(input_1):
  return {'outputs': model(input_1, training=True)}

model.save('mod_yamnet', signatures={'serving_default': model_train})

model.compile()
model.summary()