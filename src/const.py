import os
import numpy as np


TOP_LEVEL_PATH: str = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))
OUTPUT_PATH: str = os.path.join(TOP_LEVEL_PATH, "output")

MAX_DIMENSION: int = 512
VGG_MEAN: np.ndarray = np.float32([103.939, 116.779, 123.68])  # BGR
WEIGHT_URL: str = (
        'https://github.com/fchollet/deep-learning-models/'
        'releases/download/v0.1/'
        'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
)
