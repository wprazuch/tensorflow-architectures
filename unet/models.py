import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.python.keras.layers.convolutional import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
)


class DownsampleBlock(Model):
    def __init__(self, filter_sizes):
        super(DownsampleBlock, self).__init__()

        self.conv1 = Conv2D(
            filter=filter_sizes[0],
            kernel_size=(3, 3),
            strides=1,
            padding="same",
        )
        self.bn1 = BatchNormalization()

        self.act1 = Activation("relu")

        self.conv2 = Conv2D(
            filter=filter_sizes[1],
            kernel_size=(3, 3),
            strides=1,
            padding="same",
        )

        self.bn2 = BatchNormalization()

        self.act2 = Activation("relu")

        self.mp = MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        out = self.act2(x)

        return out


class UpsampleBlock(Model):
    def __init__(self, filters):
        super(UpsampleBlock, self).__init__()

        self.deconv = Conv2DTranspose(
            filter=filters[0], kernel_size=(3, 3), strides=1, padding="same"
        )

        self.concat = Concatenate()

        self.conv1 = Conv2D(
            filter=filters[1],
            kernel_size=(3, 3),
            strides=1,
            padding="same",
        )
        self.bn1 = BatchNormalization()

        self.act1 = Activation("relu")

        self.conv2 = Conv2D(
            filter=filters[2],
            kernel_size=(3, 3),
            strides=1,
            padding="same",
        )

        self.bn2 = BatchNormalization()

        self.act2 = Activation("relu")

    def call(self, inputs):

        upsampled = self.deconv(inputs[0])
        concatenated = self.concat([inputs[1], upsampled])

        x = self.conv1(concatenated)
        x = self.bn1(concatenated)
        x = self.act1(x)

        x = self.conv2(concatenated)
        x = self.bn2(concatenated)
        out = self.act2(x)

        return out


class UNet(Model):
    def __init__(self):
        super(UNet, self).__init__()

        self.downsample1 = DownsampleBlock((64, 64))
        self.mp1 = MaxPooling2D()

        self.downsample2 = DownsampleBlock((128, 128))
        self.mp2 = MaxPooling2D()

        self.downsample3 = DownsampleBlock((256, 256))
        self.mp3 = MaxPooling2D()

        self.downsample4 = DownsampleBlock((512, 512))
        self.mp4 = MaxPooling2D()
