import numpy as np
import tensorflow as tf

from tensorflow.keras import Input, Model, Sequential
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


class DoubleConvBlock(Model):
    def __init__(self, filters):
        super(DoubleConvBlock, self).__init__()

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
    def __init__(self, input_shape, num_classes: int = 10):
        self.inp_shape = input_shape
        self.num_classes = num_classes
        self.build_model()

    def build_model(self):
        self.inp = Input(input_shape=self.input_shape)

        self.down1 = DoubleConvBlock([64, 64])(self.inp)
        self.mp1 = MaxPooling2D()(self.down1)

        self.down2 = DoubleConvBlock([128, 128])(self.mp1)
        self.mp2 = MaxPooling2D()(self.down2)

        self.down3 = DoubleConvBlock([256, 256])(self.mp2)
        self.mp3 = MaxPooling2D()(self.down3)

        self.down4 = DoubleConvBlock([512, 512])(self.mp3)
        self.mp4 = MaxPooling2D()(self.down4)

        self.down5 = DoubleConvBlock([1024, 1024])(self.mp4)

        self.deconv4 = Conv2DTranspose(filters=512, kernel_size=(2, 2), padding="same")(
            self.down5
        )
        self.concat4 = Concatenate()([self.deconv4, self.down4])
        self.up4 = DoubleConvBlock([512, 512])(self.concat4)

        self.deconv3 = Conv2DTranspose(filters=512, kernel_size=(2, 2), padding="same")(
            self.up4
        )
        self.concat3 = Concatenate()([self.deconv3, self.down3])
        self.up3 = DoubleConvBlock([512, 512])(self.concat3)

        self.deconv2 = Conv2DTranspose(filters=512, kernel_size=(2, 2), padding="same")(
            self.up3
        )
        self.concat2 = Concatenate()([self.deconv2, self.down2])
        self.up2 = DoubleConvBlock([512, 512])(self.concat2)

        self.deconv1 = Conv2DTranspose(filters=512, kernel_size=(2, 2), padding="same")(
            self.up2
        )
        self.concat1 = Concatenate()([self.deconv1, self.down1])
        self.up1 = DoubleConvBlock([512, 512])(self.concat1)

        self.out = Conv2D(1, kernel_size=(1, 1), activation="sigmoid")(self.up1)

        self.model = Model(inputs=[self.inp], outputs=[self.out])

    def call(self, inputs, training=False):

        return self.model(inputs)
