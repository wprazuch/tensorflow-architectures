import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.normalization import BatchNormalization


def downsample(filters, kernel_size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = Sequential()
    result.add(
        Conv2D(
            filters,
            kernel_size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return result