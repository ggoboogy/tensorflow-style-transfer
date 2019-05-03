import numpy as np
from tensorflow.python.keras.utils import data_utils as keras_utils
from tensorflow.python.keras import models as keras_models
from tensorflow.python.keras import layers as keras_layers
from typing import Tuple
from . import const


class VGG19:
    def __init__(self) -> None:
        self.layers = [
                ('conv', 'block1_conv1', 64),
                ('conv', 'block1_conv2', 64),
                ('conv', 'block2_conv1', 128),
                ('conv', 'block2_conv2', 128),
                ('conv', 'block3_conv1', 256),
                ('conv', 'block3_conv2', 256),
                ('conv', 'block3_conv3', 256),
                ('conv', 'block3_conv4', 256),
                ('conv', 'block4_conv1', 512),
                ('conv', 'block4_conv2', 512),
                ('conv', 'block4_conv3', 512),
                ('conv', 'block4_conv4', 512),
                ('conv', 'block5_conv1', 512),
                ('conv', 'block5_conv2', 512),
                ('conv', 'block5_conv3', 512),
                ('conv', 'block5_conv4', 512)
        ]

    def _get_weight(self) -> str:
        return keras_utils.get_file(
                const.WEIGHT_URL[-1],
                const.WEIGHT_URL,
                cache_subdir='models'
        )

    def _get_conv_layer(self, filters: int, kernel_size: Tuple[int, int],
            name: str) -> keras_layers.Conv2D:
        return keras_layers.Conv2D(
                filters,
                kernel_size,
                activation='relu',
                padding='same',
                name=name
        )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # normalization by mean
        return image - const.VGG_MEAN

    def build_model(self) -> keras_models.Model:
        input_shape = keras_layers.Input(shape=(None, None, 3))
        kernel_size = (3, 3)

        x = input_shape
        for type_, name, filters in self.layers:
            if type_ == 'conv':
                x = self._get_conv_layer(filters, kernel_size, name)(x)
            else:
                pass
                # TODO max pooling

        # create model
        model = keras_models.Model(input_shape, x, name='vgg19')

        # load weight
        weight_path = self._get_weight()
        model.load_weights(weight_path)

        return model
