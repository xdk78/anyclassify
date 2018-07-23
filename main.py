import tensorflow as tf
from tensorflow import keras

# 0 - anime, 1 - human
lables = ['anime', 'human']
model = keras.Sequential([
    keras.layers.Conv2D(
        20, (5, 5),
        padding='same',
        input_shape=(3, 100, 100)),  # 3 for RGB, image 100 x 100, we need to convert 2D to 1D
    keras.layers.Dense(
        units=128,  # hidden neurons
        activation=tf.nn.relu),
    keras.layers.Dense(
        units=10,  # output neurons
        activation=tf.nn.softmax)
])
print(model.input_shape)
print(model.output_shape)
# sess = tf.Session()
# print(sess.run())
