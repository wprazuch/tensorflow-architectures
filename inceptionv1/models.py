import tensorflow as tf
from tensorflow.keras import Model, Sequential


from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
)


def inception_block(
    input_tensor, branch_1_filter, branch_2_filters, branch_3_filters, branch_4_filter
):

    branch_1_1 = Conv2D(
        filters=branch_1_filter, kernel_size=(1, 1), strides=1, padding="same"
    )(input_tensor)

    branch_2_1 = Conv2D(
        filters=branch_2_filters[0], kernel_size=(1, 1), strides=1, padding="same"
    )(input_tensor)
    branch_2_2 = Conv2D(
        filters=branch_2_filters[1], kernel_size=(3, 3), strides=1, padding="same"
    )(branch_2_1)

    branch_3_1 = Conv2D(
        filters=branch_3_filters[0], kernel_size=(1, 1), strides=1, padding="same"
    )(input_tensor)
    branch_3_2 = Conv2D(
        filters=branch_3_filters[1], kernel_size=(5, 5), strides=1, padding="same"
    )(branch_3_1)

    branch_4_1 = MaxPooling2D(pool_size=(3, 3))(input_tensor)
    branch_4_2 = Conv2D(
        filters=branch_4_filter, kernel_size=(1, 1), strides=1, padding="same"
    )(branch_4_1)

    output = Concatenate([branch_1_1, branch_2_2, branch_3_2, branch_4_2])

    return output


class InceptionBlock(Model):
    def __init__(
        self, branch_1_filter, branch_2_filters, branch_3_filters, branch_4_filter
    ):

        super(InceptionBlock, self).__init__()

        self.branch_1_1 = Conv2D(
            filters=branch_1_filter,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            activation="relu",
        )

        self.branch_2_1 = Conv2D(
            filters=branch_2_filters[0],
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            activation="relu",
        )
        self.branch_2_2 = Conv2D(
            filters=branch_2_filters[1],
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="relu",
        )

        self.branch_3_1 = Conv2D(
            filters=branch_3_filters[0],
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            activation="relu",
        )
        self.branch_3_2 = Conv2D(
            filters=branch_3_filters[1],
            kernel_size=(5, 5),
            strides=1,
            padding="same",
            activation="relu",
        )

        self.branch_4_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")
        self.branch_4_2 = Conv2D(
            filters=branch_4_filter,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            activation="relu",
        )

    def call(self, inputs, training=False):

        x1 = self.branch_1_1(inputs)

        x2 = self.branch_2_1(inputs)
        x2 = self.branch_2_2(x2)

        x3 = self.branch_3_1(inputs)
        x3 = self.branch_3_2(x3)

        x4 = self.branch_4_1(inputs)
        x4 = self.branch_4_2(x4)

        out = Concatenate(axis=-1)([x1, x2, x3, x4])

        return out


class InceptionV1(Model):
    def __init__(self):
        super(InceptionV1, self).__init__()

        self.conv1 = Conv2D(
            filters=64, kernel_size=(2, 2), strides=1, padding="same", activation="relu"
        )

        self.mp1 = MaxPooling2D(pool_size=(2, 2), strides=1)

        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(
            192, kernel_size=(1, 1), strides=1, padding="same", activation="relu"
        )

        self.mp2 = MaxPooling2D(pool_size=(2, 2), strides=1)

        self.inception3a = InceptionBlock(
            branch_1_filter=64,
            branch_2_filters=[96, 128],
            branch_3_filters=[16, 32],
            branch_4_filter=32,
        )

        self.inception3b = InceptionBlock(
            branch_1_filter=128,
            branch_2_filters=[128, 192],
            branch_3_filters=[32, 96],
            branch_4_filter=64,
        )

        self.mp3 = MaxPooling2D(pool_size=(3, 3), strides=2)

        self.inception4a = InceptionBlock(
            branch_1_filter=192,
            branch_2_filters=[96, 208],
            branch_3_filters=[16, 48],
            branch_4_filter=64,
        )

        self.inception4b = InceptionBlock(
            branch_1_filter=160,
            branch_2_filters=[112, 224],
            branch_3_filters=[24, 64],
            branch_4_filter=64,
        )

        self.inception4c = InceptionBlock(
            branch_1_filter=128,
            branch_2_filters=[128, 256],
            branch_3_filters=[24, 64],
            branch_4_filter=64,
        )

        self.inception4d = InceptionBlock(
            branch_1_filter=112,
            branch_2_filters=[144, 288],
            branch_3_filters=[32, 64],
            branch_4_filter=64,
        )

        self.inception4e = InceptionBlock(
            branch_1_filter=256,
            branch_2_filters=[160, 320],
            branch_3_filters=[32, 128],
            branch_4_filter=128,
        )

        self.mp4 = MaxPooling2D(pool_size=(3, 3), strides=2)

        self.inception5a = InceptionBlock(
            branch_1_filter=256,
            branch_2_filters=[160, 320],
            branch_3_filters=[32, 128],
            branch_4_filter=128,
        )

        self.inception5b = InceptionBlock(
            branch_1_filter=384,
            branch_2_filters=[192, 384],
            branch_3_filters=[48, 128],
            branch_4_filter=128,
        )

        self.avg_pool = GlobalAveragePooling2D()

        self.dropout = Dropout(0.4)

        self.dense = Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.mp1(x)

        x = self.bn1(x)
        x = self.conv2(x)
        x = self.mp2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.mp3(x)

        x = self.inception4a(x)
        # out1 =

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        # out2 =

        x = self.inception4e(x)
        x = self.mp4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avg_pool(x)
        x = self.dropout(x)
        out3 = self.dense(x)

        return out3
