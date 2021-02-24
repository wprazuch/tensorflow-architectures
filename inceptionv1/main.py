import os

import tensorflow as tf
from models import InceptionV1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
input_shape = (None, 32, 32, 3)

model = InceptionV1()
model.build(input_shape=input_shape)
print(model.summary())
optimizer = Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)


model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=32)
print("Done")