from numpy.core.fromnumeric import shape
import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Concatenate, Conv2D, Dropout, Input
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU, ReLU
from tensorflow.python.keras.layers.convolutional import ZeroPadding2D
from tensorflow.python.keras.layers.normalization import BatchNormalization

import config


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


def upsample(filters, size, apply_dropout=False):

    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()

    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    result.add(BatchNormalization())

    if apply_dropout:
        result.add(Dropout(0.5))

    result.add(ReLU())

    return result


def Generator():

    inputs = Input(shape=[config.IMG_WIDTH, config.IMG_HEIGHT, config.INPUT_CHANNELS])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        config.OUTPUT_CHANNELS,
        4,
        strides=2,
        kernel_initializer=initializer,
        activation="tanh",
    )

    x = inputs

    skips = []

    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])

    x = last(x)

    return Model(inputs=inputs, outputs=x)


def Discriminator():

    initializer = tf.random_normal_initializer(0.0, 0.02)

    inputs = Input(
        shape=[config.IMG_WIDTH, config.IMG_HEIGHT, config.INPUT_CHANNELS],
        name="input_image",
    )

    target = Input(
        shape=[config.IMG_WIDTH, config.IMG_HEIGHT, config.INPUT_CHANNELS],
        name="target_image",
    )

    x = Concatenate()([inputs, target])

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = ZeroPadding2D()

    conv = Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(
        zero_pad1
    )

    batchnorm1 = BatchNormalization()(conv)

    leaky_relu = LeakyReLU()(batchnorm1)

    zero_pad2 = ZeroPadding2D()(leaky_relu)

    last = Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

    return Model(inputs=[inputs, target], outputs=last)
