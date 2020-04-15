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
from tensorflow.keras import models
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.utils import normalize

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

samples = []

model = models.load_model('model.h5')

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
  samples.append(tp9)
  samples.append(af7)
  samples.append(af8)
  samples.append(tp10)
  if (calls > 127):
    last_time = time.time()
    calls = 0
    sample_file_count += 1
    samples = [samples]
    data = np.array(samples)
    data = data.reshape((1, 512))
    data = normalize(data)
    data = np.expand_dims(data, axis=2)
    samples = []
    print("PREDICT: " + str(model.predict(data)))

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