import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pathlib
import time
import random

def getPaths(name):
    imgs = []
    for filepath in pathlib.Path("dataset/"+name).glob('**/*'):
        imgs.append(filepath.absolute())
    return imgs

dataPaths = (getPaths("anime") + getPaths("faces"))
random.shuffle(dataPaths)
train_images = []
# 0 - anime, 1 - human
train_labels = []

for currentPath in dataPaths:
    im = keras.preprocessing.image.load_img(currentPath, target_size=(100, 100))
    img = keras.preprocessing.image.img_to_array(im)
    train_images.append(img)
    label = 0
    if "face" in str(currentPath):
        label = 1
    train_labels.append(label)


print(len(train_images))
print(len(train_labels))

model = keras.Sequential([
    keras.layers.Conv2D(
        20, (5, 5),
        padding='same',
        # 3 for RGB, image 100 x 100, we need to convert 2D to 1D
        input_shape=(100, 100, 3),
        activation=tf.nn.relu),
    keras.layers.MaxPooling2D(
        pool_size=(2, 2)),
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
    loss='binary_crossentropy',
    metrics=['accuracy'])

model.fit(np.array(train_images, dtype="float") / 255.0, np.asarray(train_labels), epochs = 5)
im = keras.preprocessing.image.load_img("testdata/anime_1.jpg", target_size=(100, 100))
img = keras.preprocessing.image.img_to_array(im)

test_images = [img]
# 0 - anime, 1 - human
test_labels = [[0]]

test_loss, test_acc = model.evaluate(np.array(test_images, dtype="float") / 255.0, np.asarray(test_labels),)

print('Test accuracy:', test_acc)
predictions = model.predict(np.array(test_images, dtype="float") / 255.0,)
print(predictions)
