from keras.models import Sequential
from scipy import ndimage, misc
import numpy as np
import os
import cPickle as pickle
import pandas as pd

import h5py

from keras.layers import Dense, Activation
from keras.regularizers import l2, activity_l2

model = Sequential()
model.add(Dense(32, input_shape=(12288,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(output_dim=1))

model.load_weights('ava_simple.h5')
model.compile(loss='mean_squared_error', optimizer='adam')

X = pickle.load( open("images.p", "rb"))

print model.predict(X[0].reshape(1,12288))

filepath = os.path.join(os.getcwd(), "forest.jpg")
image = ndimage.imread(filepath, mode="RGB")
image_resized = misc.imresize(image, (64, 64))
print model.predict(image_resized.reshape(1,12288))