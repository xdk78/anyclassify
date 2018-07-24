import tensorflow as tf
from tensorflow import keras
import numpy as np

maxlen = 64
chars = 0  # add chars

model = keras.Sequential([
    keras.layers.LSTM(128, input_shape=(maxlen, len(chars))),
    keras.layers.Dense(len(chars)),
    keras.layers.Dense(units=256, activation=tf.nn.softmax)
])

model.compile(
    optimizer=keras.optimizers.RMSprop(lr=0.01),
    loss='categorical_crossentropy')
