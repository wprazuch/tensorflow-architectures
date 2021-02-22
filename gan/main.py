import os
from numpy.core.records import array

from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy
import cv2

from models import Discriminator, Generator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 100
IMG_SHAPE = (28, 28, 1)

(X_train, _), (_, _) = mnist.load_data()

NO_ITERS = X_train.shape[0] // BATCH_SIZE

X_train = X_train / 127.5 - 1.0
X_train = np.expand_dims(X_train, axis=-1)

valid = np.ones((BATCH_SIZE, 1))
fake = np.zeros((BATCH_SIZE, 1))

generator = Generator(IMG_SHAPE)
discriminator = Discriminator(IMG_SHAPE)

discriminator_loss = BinaryCrossentropy(from_logits=False)

gen_optimizer = Adam(learning_rate=1e-3)
disc_optimizer = Adam(learning_rate=1e-4)

for i in tqdm(range(EPOCHS)):

    for _ in tqdm(range(NO_ITERS)):

        img_idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
        rand_noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))

        real_images = X_train[img_idx, ...]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            generated_images = generator(rand_noise)

            generated_images_preds = discriminator(generated_images)
            real_images_preds = discriminator(real_images)

            total_disc_loss = (
                discriminator_loss(valid, real_images_preds)
                + discriminator_loss(fake, generated_images_preds)
            ) / 2

            gen_loss = discriminator_loss(valid, generated_images_preds)

        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_gradients = disc_tape.gradient(
            total_disc_loss, discriminator.trainable_variables
        )

        gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        disc_optimizer.apply_gradients(
            zip(disc_gradients, discriminator.trainable_variables)
        )

        generated_images = generator(rand_noise)
        generated_images = generated_images.numpy()

    for j in range(5):

        arr = generated_images[j, ...]
        new_arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype(
            "uint8"
        )

        cv2.imwrite(
            f"/home/wprazuch/Projects/tensorflow-architectures/gan/generated/{i}_epoch_{j}_image.png",
            new_arr,
        )
