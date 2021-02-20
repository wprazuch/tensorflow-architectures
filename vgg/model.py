import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import datasets, layers, optimizers, models
from tensorflow.keras import regularizers

from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)

from tensorflow.keras import Model
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.ops.gen_math_ops import xdivy


def VGG16(input_shape):

    inputs = Input(shape=input_shape)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(
        inputs
    )
    x = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D()(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D()(x)
    x = Flatten()(x)

    x = Dense(4096, activation="relu")(x)
    x = Dense(4096, activation="relu")(x)
    x = Dense(10, activation="softmax")(x)
    return Model(inputs=inputs, outputs=x)
