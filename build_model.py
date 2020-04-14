"""Small example OSC server

This program listens to several addresses, and prints some information about
received packets.
"""
import argparse
import math

from pythonosc import dispatcher
from pythonosc import osc_server

import time

import numpy as np
import threading
import os
import tensorflow as tf

# These match with output neuron activation values
LABEL_GREEN = 0
LABEL_RED = 1
LABEL_CONTROL = 2

OUTPUTS = [[1.0, 0.0],
  [0.0, 1.0],
  [0.0, 0.0]]

def get_samples_and_outputs(ddir, label):
  samples = []
  #samples = np.empty([0, 4])
  outputs = []

  files = os.listdir(ddir)
  for file in files:
    data = np.fromfile(ddir + file)
    samples.append(data)
    outputs.append(OUTPUTS[label])

  return (np.asarray(samples), np.asarray(outputs))

(g1_data, g1_outputs) = get_samples_and_outputs("data/green1/", LABEL_GREEN)
(g2_data, g2_outputs) = get_samples_and_outputs("data/green2/", LABEL_GREEN)
(r1_data, r1_outputs) = get_samples_and_outputs("data/red1/", LABEL_RED)
(r2_data, r2_outputs) = get_samples_and_outputs("data/red2/", LABEL_RED)
(c1_data, c1_outputs) = get_samples_and_outputs("data/neither1/", LABEL_CONTROL)
(c2_data, c2_outputs) = get_samples_and_outputs("data/neither2/", LABEL_CONTROL)
(c3_data, c3_outputs) = get_samples_and_outputs("data/neither3/", LABEL_CONTROL)

print(g1_data)
print(g1_outputs)

train_dataset = tf.data.Dataset.from_tensor_slices((g2_data, g2_outputs))

'''import numpy as np
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Reshape the images.
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

num_filters = 8
filter_size = 3
pool_size = 2

# Build the model.
model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(10, activation='softmax'),
])

# Compile the model.
model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the model.
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=3,
  validation_data=(test_images, to_categorical(test_labels)),
)

# Save the model to disk.
model.save_weights('cnn.h5')

# Load the model from disk later using:
# model.load_weights('cnn.h5')

# Predict on the first 5 test images.
predictions = model.predict(test_images[:5])

# Print our model's predictions.
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print(test_labels[:5]) # [7, 2, 1, 0, 4]'''