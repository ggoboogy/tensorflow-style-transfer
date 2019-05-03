import argparse
import tensorflow as tf
import numpy as np
from . import util
from .vgg19 import VGG19


def _gram_matrix(tensor: np.ndarray) -> tf.Tensor:
    channel_cnt = tensor.shape[-1]
    # (r, c, channel_cnt) -> (r * c, channel_cnt)
    matrix = tf.reshape(tensor, shape=[-1, channel_cnt])
    # transpose row and column, and then conduct matrix multiplication
    # (channel_cnt, r * c) * (r * c, channel_cnt) -> (channel_cnt, channel_cnt)
    gram = tf.matmul(matrix, matrix, transpose_a=True)
    return gram


def _mean_squared_error(predicted: np.ndarray,
        actual: np.ndarray) -> np.ndarray:
    return tf.reduce_mean(tf.square(predicted - actual))


def _style_transfer(input_path: str, style_path: str, output_path: str):
    # load input/style image
    input_, style = util.load_image(input_path), util.load_image(style_path)

    # preprocess images and build model
    vgg = VGG19()
    input_, style = vgg.preprocess(input_), vgg.preprocess(style)
    _ = vgg.build_model()


def main() -> None:
    parser = argparse.ArgumentParser(
            description="""
            Style Transfer
            """, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
            "--input",
            dest="input",
            required=True,
            type=util.check_input_existance,
            help="TODO"
    )
    parser.add_argument(
            "--style",
            dest="style",
            required=True,
            type=util.check_input_existance,
            help="TODO"
    )
    parser.add_argument(
            "--output",
            dest="output",
            required=True,
            type=util.check_output_existance,
            help="TODO"
    )
    args = parser.parse_args()

    _style_transfer(args.input, args.style, args.output)
