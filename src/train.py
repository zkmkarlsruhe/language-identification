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


def train(config_path, log_dir, model_path):

    # Config
    config = load(open(config_path, "rb"))
    if config is None:
        print("Please provide a config.")

    train_dir = config["train_dir"]
    val_dir = config["val_dir"]
    batch_size = config["batch_size"]
    img_height = config["input_shape"][0]
    img_width = config["input_shape"][1]
    color_mode = config["color_mode"]
    seed = config["seed"]

    # Dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=seed,
        label_mode="categorical",
        color_mode=color_mode,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        seed=seed,
        label_mode="categorical",
        color_mode=color_mode,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True)

    if color_mode == "rgba":
        num_channels = 4
    elif color_mode == "rgb":
        num_channels = 3
    else:
        num_channels = 1

    ts_shape = (batch_size, img_height, img_width, num_channels)

    print("Input image shape:", ts_shape[1:])
    print("Output classes:", train_ds.class_names)

    # normalize images
    def process(image, label):
        image = tf.cast(image / 255., tf.float32)
        return image, label

    train_ds = train_ds.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # prefetch data
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

