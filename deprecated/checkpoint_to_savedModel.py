# ofxTensorFlow2
#
# Copyright (c) 2021 ZKM | Hertz-Lab
# Paul Bethge <bethge@zkm.de>
#
# BSD Simplified License.
# For information on usage and redistribution, and for a DISCLAIMER OF ALL
# WARRANTIES, see the file, "LICENSE.txt," in this distribution.
#
# This code has been developed at ZKM | Hertz-Lab as part of „The Intelligent 
# Museum“ generously funded by the German Federal Cultural Foundation.


import tensorflow as tf
import os
import argparse

import trained_models.models as models

def convert(src_dir, dest_dir):

	config = load(open(config_path, "rb"))
	if config is None:
		print("Please provide a config.")

	input_height = config["input_shape"][0]
	input_width = config["input_shape"][1]
	model_name = config["model"]

    # Build the feed-forward network and load the weights.
	model_class = getattr(models, self._model_name)
	self._model = model_class.create_model(config)
	self._model.load_weights(model_path+"variables/variables")
    network.load_weights(os.path.join(src_dir,"weights")).expect_partial()

    model_name = os.path.join(dest_dir, src_dir.split('/')[-1])

    if not os.path.exists(src_dir):
        os.makedirs(src_dir)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    network.build(tuple(shape))

    @tf.function(input_signature=[tf.TensorSpec(shape, dtype=tf.float32)])
    def model_predict(input_1):
        return {'outputs': network(input_1, training=False)}

    network.save(model_name, signatures={'serving_default': model_predict})


if __name__ == "__main__":

    dest = os.path.dirname('./savedModel/')

	config_path = "trained_models/config.yaml"
	weights_path = "trained_models/weights.18"

	convert(config_path, weights_path, dest)

