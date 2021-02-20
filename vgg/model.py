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


def VGG16Keras(
    input_shape=None,
    classes=10,
):

    img_input = layers.Input(shape=input_shape)
    inputs = img_input

    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv1"
    )(inputs)
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1_conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2_conv1"
    )(x)
    x = layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2_conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv1"
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv2"
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3_conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv1"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv2"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4_conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5_conv1"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5_conv2"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5_conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(4096, activation="relu", name="fc1")(x)
    x = layers.Dense(4096, activation="relu", name="fc2")(x)
    x = layers.Dense(classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, x, name="vgg16")

    return model


class SmallNet(Model):
    def __init__(self):
        super(SmallNet, self).__init__()

        self.c1 = Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.mp1 = MaxPooling2D()
        self.c2 = Conv2D(128, kernel_size=(3, 3), activation="relu")
        self.mp2 = MaxPooling2D()
        self.c3 = Conv2D(256, kernel_size=(3, 3), activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(10, activation="softmax")

    def call(self, x):
        x = self.c1(x)
        x = self.mp1(x)
        x = self.c2(x)
        x = self.mp2(x)
        x = self.c3(x)
        x = self.flatten(x)
        x = self.d1(x)
        return x


class VGG16(Model):
    def __init__(self):
        super(VGG16, self).__init__()

        weight_decay = 0.000
        self.num_classes = 10

        self.conv1 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(
            filters=64, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.bn2 = BatchNormalization()

        self.mp1 = MaxPooling2D()
        self.conv3 = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn3 = BatchNormalization()
        self.conv4 = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn4 = BatchNormalization()

        self.mp2 = MaxPooling2D()
        self.conv5 = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn5 = BatchNormalization()
        self.conv6 = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn6 = BatchNormalization()
        self.conv7 = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn7 = BatchNormalization()

        self.mp3 = MaxPooling2D()
        self.conv8 = Conv2D(
            filters=512,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn8 = BatchNormalization()
        self.conv9 = Conv2D(
            filters=512,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn9 = BatchNormalization()
        self.conv10 = Conv2D(
            filters=512,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn10 = BatchNormalization()

        self.mp4 = MaxPooling2D()
        self.conv11 = Conv2D(
            filters=512,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn11 = BatchNormalization()
        self.conv12 = Conv2D(
            filters=512,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn12 = BatchNormalization()
        self.conv13 = Conv2D(
            filters=512,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay),
        )
        self.bn13 = BatchNormalization()

        self.mp5 = MaxPooling2D()
        self.flatten = Flatten()

        self.fc1 = Dense(4096, activation="relu")
        self.fc2 = Dense(4096, activation="relu")
        self.fc3 = Dense(10, activation="relu")

    def call(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mp1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.mp2(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.mp3(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.mp4(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.mp5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x