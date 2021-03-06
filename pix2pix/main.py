import tensorflow as tf

import os
import time

from matplotlib import pyplot as plt

from utils import load, random_jitter, load_image_train, load_image_test
import config


_URL = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz"

path_to_zip = tf.keras.utils.get_file("facades.tar.gz", origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), "facades/")


inp, re = load(PATH + "train/100.jpg")
# casting to int for matplotlib to show the image
plt.figure()
plt.imshow(inp / 255.0)
plt.figure()
plt.imshow(re / 255.0)

plt.figure(figsize=(6, 6))
for i in range(4):
    rj_inp, rj_re = random_jitter(inp, re)
    plt.subplot(2, 2, i + 1)
    plt.imshow(rj_inp / 255.0)
    plt.axis("off")
plt.show()


train_dataset = tf.data.Dataset.list_files(PATH + "train/*.jpg")
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(config.BUFFER_SIZE)
train_dataset = train_dataset.batch(config.BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH + "test/*.jpg")
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(config.BATCH_SIZE)
