import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    GlobalAveragePooling2D,
    Dense,
    Flatten,
    MaxPooling2D,
)


class ResidualBlock(Model):
    def __init__(self, no_channels, strides):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(
            filters=no_channels,
            kernel_size=(3, 3),
            strides=strides,
            padding="same",
            activation="relu",
        )
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(
            filters=no_channels,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="relu",
        )
        self.bn2 = BatchNormalization()

        if strides != 1:
            self.residual = Sequential()
            self.residual.add(
                Conv2D(
                    filters=no_channels,
                    kernel_size=(3, 3),
                    padding="same",
                    strides=strides,
                    activation="relu",
                )
            )
            self.residual.add(BatchNormalization())
        else:
            self.residual = lambda x: x

    def call(self, inputs):

        residual = self.residual(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = residual + x

        return x


class BottleneckBlock(Model):
    def __init__(self, in_channels, out_channels, strides):
        super(BottleneckBlock, self).__init__()

        self.conv1 = Conv2D(
            filters=in_channels,
            kernel_size=(3, 3),
            strides=strides,
            padding="same",
            activation="relu",
        )
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(
            filters=in_channels,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="relu",
        )
        self.bn2 = BatchNormalization()

        self.conv3 = Conv2D(
            filters=out_channels,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="relu",
        )
        self.bn3 = BatchNormalization()

        if strides != 1:
            self.shortcut = Sequential()
            self.shortcut.add(
                Conv2D(
                    filters=out_channels,
                    kernel_size=(3, 3),
                    strides=strides,
                    padding="same",
                    activation="relu",
                )
            )
            self.shortcut.add(BatchNormalization())

        else:
            self.shortcut = lambda x: x

    def call(self, inputs):

        shortcut = self.shortcut(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        return x + shortcut


class ResNet50(Model):
    def __init__(self, input_shape):
        super(ResNet50, self).__init__()
        self.in_shape = input_shape

        self.conv1 = Conv2D(
            256,
            input_shape=input_shape,
            kernel_size=(7, 7),
            strides=2,
            padding="same",
            activation="relu",
        )
        self.bn1 = BatchNormalization()
        self.mp1 = MaxPooling2D(pool_size=(3, 3))

        self.res_block1 = BottleneckBlock(in_channels=64, out_channels=256, strides=1)
        self.res_block2 = BottleneckBlock(in_channels=64, out_channels=256, strides=1)
        self.res_block3 = BottleneckBlock(in_channels=64, out_channels=256, strides=1)

        self.res_block4 = BottleneckBlock(in_channels=128, out_channels=512, strides=2)
        self.res_block5 = BottleneckBlock(in_channels=128, out_channels=512, strides=1)
        self.res_block6 = BottleneckBlock(in_channels=128, out_channels=512, strides=1)
        self.res_block7 = BottleneckBlock(in_channels=128, out_channels=512, strides=1)

        self.res_block8 = BottleneckBlock(in_channels=256, out_channels=1024, strides=2)
        self.res_block9 = BottleneckBlock(in_channels=256, out_channels=1024, strides=1)
        self.res_block10 = BottleneckBlock(
            in_channels=256, out_channels=1024, strides=1
        )
        self.res_block11 = BottleneckBlock(
            in_channels=256, out_channels=1024, strides=1
        )
        self.res_block12 = BottleneckBlock(
            in_channels=256, out_channels=1024, strides=1
        )
        self.res_block13 = BottleneckBlock(
            in_channels=256, out_channels=1024, strides=1
        )

        self.res_block14 = BottleneckBlock(
            in_channels=512, out_channels=2048, strides=2
        )
        self.res_block15 = BottleneckBlock(
            in_channels=512, out_channels=2048, strides=1
        )
        self.res_block16 = BottleneckBlock(
            in_channels=512, out_channels=2048, strides=1
        )

        self.avg_pool = GlobalAveragePooling2D()
        self.dense = Dense(10, activation="softmax")

    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.mp1(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.res_block7(x)
        x = self.res_block8(x)
        x = self.res_block9(x)
        x = self.res_block10(x)
        x = self.res_block11(x)
        x = self.res_block12(x)
        x = self.res_block13(x)
        x = self.res_block14(x)
        x = self.res_block15(x)
        x = self.res_block16(x)

        x = self.avg_pool(x)
        x = self.dense(x)

        return x


class ResNet18(Model):
    def __init__(self, input_shape):
        super(ResNet18, self).__init__()
        self.in_shape = input_shape

        self.conv1 = Conv2D(
            64,
            input_shape=input_shape,
            kernel_size=(7, 7),
            strides=2,
            padding="same",
            activation="relu",
        )
        self.bn1 = BatchNormalization()
        self.mp1 = MaxPooling2D(pool_size=(3, 3))

        self.res_block1 = ResidualBlock(no_channels=64, strides=1)
        self.res_block2 = ResidualBlock(no_channels=64, strides=1)

        self.res_block3 = ResidualBlock(no_channels=128, strides=2)
        self.res_block4 = ResidualBlock(no_channels=128, strides=1)

        self.res_block5 = ResidualBlock(no_channels=256, strides=2)
        self.res_block6 = ResidualBlock(no_channels=256, strides=1)

        self.res_block5 = ResidualBlock(no_channels=512, strides=2)
        self.res_block6 = ResidualBlock(no_channels=512, strides=1)

        self.avg_pool = GlobalAveragePooling2D()
        self.dense = Dense(10, activation="softmax")

    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.mp1(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)

        x = self.avg_pool(x)
        x = self.dense(x)

        return x


class ResNet34(Model):
    def __init__(self, input_shape):
        super(ResNet34, self).__init__()
        self.in_shape = input_shape

        self.conv1 = Conv2D(
            64,
            input_shape=input_shape,
            kernel_size=(7, 7),
            strides=2,
            padding="same",
            activation="relu",
        )
        self.bn1 = BatchNormalization()
        self.mp1 = MaxPooling2D(pool_size=(3, 3))

        self.res_block1 = ResidualBlock(no_channels=64, strides=1)
        self.res_block2 = ResidualBlock(no_channels=64, strides=1)
        self.res_block3 = ResidualBlock(no_channels=64, strides=1)

        self.res_block4 = ResidualBlock(no_channels=128, strides=2)
        self.res_block5 = ResidualBlock(no_channels=128, strides=1)
        self.res_block6 = ResidualBlock(no_channels=128, strides=1)
        self.res_block7 = ResidualBlock(no_channels=128, strides=1)

        self.res_block8 = ResidualBlock(no_channels=256, strides=2)
        self.res_block9 = ResidualBlock(no_channels=256, strides=1)
        self.res_block10 = ResidualBlock(no_channels=256, strides=1)
        self.res_block11 = ResidualBlock(no_channels=256, strides=1)
        self.res_block12 = ResidualBlock(no_channels=256, strides=1)
        self.res_block13 = ResidualBlock(no_channels=256, strides=1)

        self.res_block14 = ResidualBlock(no_channels=512, strides=2)
        self.res_block15 = ResidualBlock(no_channels=512, strides=1)
        self.res_block16 = ResidualBlock(no_channels=512, strides=1)

        self.avg_pool = GlobalAveragePooling2D()
        self.dense = Dense(10, activation="softmax")

    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.mp1(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.res_block7(x)
        x = self.res_block8(x)
        x = self.res_block9(x)
        x = self.res_block10(x)
        x = self.res_block11(x)
        x = self.res_block12(x)
        x = self.res_block13(x)
        x = self.res_block14(x)
        x = self.res_block15(x)
        x = self.res_block16(x)

        x = self.avg_pool(x)
        x = self.dense(x)

        return x