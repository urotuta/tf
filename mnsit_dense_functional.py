import numpy as np
import tensorflow as tf
from tensorflow import keras


mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

input_shape = np.shape(x_train[0])
print("Input shape: ", input_shape)

# Define network
inputs = keras.layers.Input(input_shape)
x = keras.layers.Flatten()(inputs)

x = keras.layers.Dense(64)(x)
x = keras.layers.ReLU()(x)

x = keras.layers.Dense(128, activation='sigmoid')(x)
x = keras.layers.Dropout(0.2)(x)

outputs = keras.layers.Dense(10, activation='softmax')(x)

# Create model
model = keras.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
