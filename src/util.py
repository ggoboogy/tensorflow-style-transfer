import os, argparse
import numpy as np
from PIL import Image
from tensorflow.python.keras.preprocessing import image as keras_image
from . import const


def check_input_existance(file_name: str) -> str:
    path = os.path.join(const.TOP_LEVEL_PATH, file_name)
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"input file '{file_name}' does not exist.")


def check_output_existance(file_name: str) -> str:
    path = os.path.join(const.OUTPUT_PATH, file_name)
    if os.path.exists(path):
        raise argparse.ArgumentTypeError(
            f"output file '{file_name}' already exists.")
    else:
        return path


def load_image(path: str) -> np.ndarray:
    image = Image.open(path)
    scale = const.MAX_DIMENSION / max(image.size)
    image = image.resize(
            (round(image.size[0] * scale), round(image.size[1] * scale)),
            Image.ANTIALIAS
    )
    image = keras_image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image[..., ::-1]  # RGB -> BGR

    return np.float32(image)


def save_image(image: np.ndarray, file_name: str) -> None:
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    with open(file_name, 'wb') as f:
        Image.fromarray(image[0]).save(f, 'jpeg')
