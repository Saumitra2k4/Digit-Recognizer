import numpy as np
import os
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(X_train , Y_train),(X_test , Y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(256, activation = 'relu'))
model.add(tf.keras.layers.Dense(256, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))


model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy' , metrics=['accuracy'])
model.fit(X_train , Y_train, epochs=150)
model.save('digit_recognizer.h5')

model = tf.keras.models.load_model('digit_recognizer.h5')
loss , accuracy = model.evaluate(X_test,Y_test)
print(loss)
print(accuracy)


