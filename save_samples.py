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

last_time = 0
calls = 0

samples = np.empty([0, 4])

files = os.listdir("data/")
sample_file_count = len(files)
print(sample_file_count)

def print_volume_handler(unused_addr, args, volume):
  print("[{0}] ~ {1}".format(args[0], volume))

def print_compute_handler(unused_addr, args, volume):
  try:
    print("[{0}] ~ {1}".format(args[0], args[1](volume)))
  except ValueError: pass

def handle_sample(unused_addr, tp9, af7, af8, tp10, aux):
  global last_time
  global calls
  global samples
  global sample_file_count
  calls += 1
  samples = np.append(samples, [[tp9, af7, af8, tp10]], 0)
  if (calls > 127):
    last_time = time.time()
    samples.tofile('data/test-' + str(sample_file_count) + '.num')
    samples = np.empty([0, 4])
    calls = 0
    print("saving")
    sample_file_count += 1

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ip",
      default="192.168.1.253", help="The ip to listen on")
  parser.add_argument("--port",
      type=int, default=5005, help="The port to listen on")
  args = parser.parse_args()

  dispatcher = dispatcher.Dispatcher()
  dispatcher.map("/muse/eeg", handle_sample)

  server = osc_server.BlockingOSCUDPServer(
      (args.ip, args.port), dispatcher)
  print("Serving on {}".format(server.server_address))

  t = threading.Thread(name='daemon', target=server.serve_forever)
  t.setDaemon(True)

  t.start()

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, aspect='equal')

ax2.add_patch(
     patches.Rectangle(
        (0.0, 0.0),
        1.0,
        1.0,
        fill=True,
        edgecolor='r',
        facecolor='r'
))
plt.axis('off')
plt.show()

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