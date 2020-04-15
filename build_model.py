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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import normalize

# These match with output neuron activation values
LABEL_GREEN = 0
LABEL_RED = 1
LABEL_CONTROL = 2

OUTPUTS = [[1, 0],
  [0, 1],
  [0, 0]]

LABELS = ["GREEN", "RED", "NONE"]

def shuffle_in_unison(a, b):
  assert len(a) == len(b)
  shuffled_a = np.empty(a.shape, dtype=a.dtype)
  shuffled_b = np.empty(b.shape, dtype=b.dtype)
  permutation = np.random.permutation(len(a))
  for old_index, new_index in enumerate(permutation):
      shuffled_a[new_index] = a[old_index]
      shuffled_b[new_index] = b[old_index]
  return shuffled_a, shuffled_b

def get_samples_and_outputs(ddir, label):
  samples = []
  #samples = np.empty([0, 4])
  outputs = []

  files = os.listdir(ddir)
  for file in files:
    data = np.fromfile(ddir + file)
    samples.append(data)
    outputs.append(label)

  return (np.asarray(samples), np.asarray(outputs))

(g1_data, g1_outputs) = get_samples_and_outputs("data/green1/", LABEL_GREEN)
(g2_data, g2_outputs) = get_samples_and_outputs("data/green2/", LABEL_GREEN)
(r1_data, r1_outputs) = get_samples_and_outputs("data/red1/", LABEL_RED)
(r2_data, r2_outputs) = get_samples_and_outputs("data/red2/", LABEL_RED)
(c1_data, c1_outputs) = get_samples_and_outputs("data/neither1/", LABEL_CONTROL)
(c2_data, c2_outputs) = get_samples_and_outputs("data/neither2/", LABEL_CONTROL)
(c3_data, c3_outputs) = get_samples_and_outputs("data/neither3/", LABEL_CONTROL)

all_data = np.concatenate((g1_data, g2_data, r1_data, r2_data, c1_data, c2_data, c3_data))
all_labels = np.concatenate((g1_outputs, g2_outputs, r1_outputs, r2_outputs, c1_outputs, c2_outputs, c3_outputs))

all_data = normalize(all_data)

(all_data, all_labels) = shuffle_in_unison(all_data, all_labels)

all_labels = to_categorical(all_labels)

training_data, test_data = all_data[100:, ...], all_data[:100, ...]
training_labels, test_labels = all_labels[100:, ...], all_labels[:100, ...]

print("SHAPE 1: " + str(training_data.shape))

training_data = np.expand_dims(training_data, axis=2)
test_data = np.expand_dims(test_data, axis=2)

print("SHAPE 2: " + str(training_data.shape))
print("SHAPE 3: " + str(training_labels.shape))

train_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_labels)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(512,1)))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(Dense(256, activation='tanh'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_dataset, epochs=10,
  verbose=1, shuffle=True)

model.evaluate(test_dataset)

model.save('model.h5')

'''import numpy as np
import mnist
from tensorflow.keras.models import Seq uential
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