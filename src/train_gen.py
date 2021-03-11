import os
import shutil
import argparse
from datetime import datetime
from yaml import load

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.models import load_model

import models
from utils.training_utils import *
from audio.generators import LIDGenerator

def train(config_path, log_dir, model_path):

    # Config
    config = load(open(config_path, "rb"))
    if config is None:
        print("Please provide a config.")

    train_dir = config["train_dir"]
    val_dir = config["val_dir"]
    batch_size = config["batch_size"]
    languages = config["languages"]

    # ts_shape = (batch_size, signal_length, num_channels)
    # print("Input shape:", ts_shape[1:])
    # print("Output classes:", train_ds.class_names)
    
    # create Generators
    train_gen_obj = LIDGenerator(source=train_dir, target_length_s=10, shuffle=True,
                            languages=languages)
    val_gen_obj = LIDGenerator(source=val_dir, target_length_s=10, shuffle=True,
                            languages=languages)
    # create Dataset
    train_ds = tf.data.Dataset.from_generator(train_gen_obj.get_generator, (tf.float32, tf.int16))
    val_ds = tf.data.Dataset.from_generator(val_gen_obj.get_generator, (tf.float32, tf.int16))
        
    # prepare dataset pipelines
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    
    # train_ds = train_ds.repeat()
    # val_ds = val_ds.repeat()

    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # Training Callbacks
    checkpoint_filename = os.path.join(log_dir, "trained_models", "weights.{epoch:02d}")
    model_checkpoint_callback = ModelCheckpoint(checkpoint_filename, save_best_only=True, verbose=1,
                                                 monitor="val_categorical_accuracy",
                                                 save_weights_only=False)
    tensorboard_callback = TensorBoard(log_dir=log_dir, write_images=True)
    csv_logger_callback = CustomCSVCallback(os.path.join(log_dir, "log.csv"))
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode="min")
    reduce_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, verbose=1, min_lr=0.000001,
                                           min_delta=0.001)

    # Model Generation
    if model_path:
        model = load_model(model_path)
    else:
        model_class = getattr(models, config["model"])
        model = model_class.create_model(config)
        optimizer = Adam(lr=config["learning_rate"])
        # optimizer = RMSprop(lr=config["learning_rate"], rho=0.9, epsilon=1e-08, decay=0.95)
        # optimizer = SGD(lr=config["learning_rate"], decay=1e-6, momentum=0.9, clipnorm=1, clipvalue=10)
        model.compile(optimizer=optimizer,
                      loss=CategoricalCrossentropy(),
                      metrics=[Recall(), Precision(), CategoricalAccuracy()]
                      )
    print(model.summary())

    # Training
    history = model.fit(x=train_ds,
                        epochs=config["num_epochs"],
                        callbacks=[model_checkpoint_callback,
                                   tensorboard_callback,
                                   csv_logger_callback,
                                   # early_stopping_callback,
                                   reduce_on_plateau
                                   ],
                        validation_data=val_ds)
    # visualize_results(history, config)

    # Do evaluation on model with best validation accuracy
    best_epoch = np.argmax(history.history["val_categorical_accuracy"])
    print("Log files: ", log_dir)
    print("Best epoch: ", best_epoch)
    return checkpoint_filename.replace("{epoch:02d}", "{:02d}".format(best_epoch))


if __name__ == "__main__": 
    tf.config.list_physical_devices('GPU')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--config', default="config.yaml")
    cli_args = parser.parse_args()

    log_dir = os.path.join("logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    print("Logging to {}".format(log_dir))

    # copy models & config for later
    shutil.copytree("models", os.path.join(log_dir, "models"))
    shutil.copy(cli_args.config, log_dir)

    model_file_name = train(cli_args.config, log_dir, cli_args.model_path)
    print("Best model at: ", model_file_name)

