import argparse
from yaml import load
import tensorflow as tf
from tensorflow.keras.models import load_model


def evaluate(config_path, model_path):

    # Config
    config = load(open(config_path, "rb"))
    if config is None:
        print("Please provide a config.")

    data_dir = config["data_dir"]
    batch_size = config["batch_size"]
    img_height = config["input_shape"][0]
    img_width = config["input_shape"][1]
    color_mode = config["color_mode"]
    seed = config["seed"]

    # load model
    model = load_model(filepath=model_path)

    # load validation set
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        seed=seed,
        label_mode="categorical",
        color_mode=color_mode,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # normalize images
    def process(image, label):
        image = tf.cast(image / 255., tf.float32)
        return image, label

    val_ds = val_ds.map(process)
    val_ds = val_ds.prefetch(tf.keras.experimental.AUTOTUNE)

    # evaluate
    metrics = model.evaluate(x=val_ds)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='trained_models/lang5_eff_acc92')
    parser.add_argument('--config', default="config.yaml")
    cli_args = parser.parse_args()

    evaluate(cli_args.config, cli_args.model)
