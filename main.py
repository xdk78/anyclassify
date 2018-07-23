import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pathlib
import time

train_images = []
# 0 - anime, 1 - human
train_labels = ['anime', 'human']

for filepath in pathlib.Path("dataset/anime").glob('**/*'):
    im = keras.preprocessing.image.load_img(
        filepath.absolute(), target_size=(100, 100))
    img = keras.preprocessing.image.img_to_array(im)
    train_images.append(img)

print(train_images)


model = keras.Sequential([
    keras.layers.Conv2D(
        20, (5, 5),
        padding='same',
        # 3 for RGB, image 100 x 100, we need to convert 2D to 1D
        input_shape=(100, 100, 3),
        activation=tf.nn.relu),
    keras.layers.MaxPooling2D(
        pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(
        units=128,
        activation=tf.nn.relu),
    keras.layers.Dense(
        units=1,  # output neurons
        activation=tf.nn.sigmoid)
])

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs = 5)

print(model.summary)
# sess = tf.Session()
# print(sess.run())
