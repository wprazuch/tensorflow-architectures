import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    LeakyReLU,
    BatchNormalization,
    Reshape,
)

import numpy as np


class Discriminator(Model):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.f1 = Flatten()
        self.d1 = Dense(512)
        self.l1 = LeakyReLU(alpha=0.2)

        self.d2 = Dense(256)
        self.l2 = LeakyReLU(alpha=0.2)

        self.d3 = Dense(1, activation="sigmoid")

    def call(self, inputs):

        x = self.f1(inputs)
        x = self.d1(x)
        x = self.l1(x)

        x = self.d2(x)
        x = self.l2(x)

        x = self.d3(x)

        return x


class Generator(Model):
    def __init__(self, input_shape):
        super(Generator, self).__init__()

        self.in_shape = input_shape

        self.s1 = Dense(256)
        self.l1 = LeakyReLU(alpha=0.2)
        self.bn1 = BatchNormalization()

        self.s2 = Dense(512)
        self.l2 = LeakyReLU(alpha=0.2)
        self.bn2 = BatchNormalization()

        self.s3 = Dense(1024)
        self.l3 = LeakyReLU(alpha=0.2)
        self.bn3 = BatchNormalization()

        self.d4 = Dense(np.prod(input_shape))
        self.r = Reshape(self.in_shape)

    def call(self, inputs):

        x = self.s1(inputs)
        x = self.l1(x)
        x = self.bn1(x)

        x = self.s2(x)
        x = self.l2(x)
        x = self.bn2(x)

        x = self.s3(x)
        x = self.l3(x)
        x = self.bn3(x)

        x = self.d4(x)
        x = self.r(x)

        return x