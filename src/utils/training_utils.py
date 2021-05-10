"""
:author:
Paul Bethge (bethge@zkm.de)
2021

:License:
This package is published under GNU GPL Version 3.
"""

import csv
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.backend import get_value


# Custom Callbacks
class CustomCSVCallback(Callback):

    def __init__(self, logging_dir):
        self._logging_dir = logging_dir
        self._counter = 0

    def on_epoch_end(self, epoch, logs={}):
        with open(self._logging_dir, mode='a') as log_file:
            log_file_writer = csv.writer(log_file, delimiter=',')
            if self._counter == 0:
                row = list(logs.keys())
                row.insert(0, "epoch")
                row.append("learning_rate")
                log_file_writer.writerow(row)
            row_vals = [round(x, 6) for x in list(logs.values())]
            row_vals.insert(0, self._counter)
            row_vals.append(round(float(
                get_value(self.model.optimizer.learning_rate)), 6))
            log_file_writer.writerow(row_vals)
        self._counter += 1


def write_csv(logging_dir, optimizer, epoch, logs={}):
        with open(logging_dir, mode='a') as log_file:
            log_file_writer = csv.writer(log_file, delimiter=',')
            if epoch == 0:
                row = list(logs.keys())
                row.insert(0, "epoch")
                row.append("learning_rate")
                log_file_writer.writerow(row)
            row_vals = [round(x, 6) for x in list(logs.values())]
            row_vals.insert(0, epoch+1)
            row_vals.append(round(float(
                get_value(optimizer.learning_rate)), 6))
            log_file_writer.writerow(row_vals)


def visualize_results(history, config):
    epochs = config["num_epochs"]


def visualize_results(history, config):
    epochs = config["num_epochs"]
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
