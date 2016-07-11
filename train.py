from keras.models import Sequential
from scipy import ndimage, misc
import numpy as np
import os
import cPickle as pickle
import pandas as pd

import h5py

from keras.layers import Dense, Activation
from keras.regularizers import l2, activity_l2

# ava_table = pd.read_table("dataset/AVA/AVA.txt", delim_whitespace=True)
ava_table = pd.read_table("dataset/AVA/AVA.txt", delim_whitespace=True,
 index_col=1, header=None)
ava_table.sort_index(inplace=True)
X = pickle.load( open("images.p", "rb"))

num_training = 9000
num_validation = 1000
X_train = np.hstack(X).reshape(10000,-1).T #.transpose(0,2,3,1).astype("float")
Y_train = ava_table.ix[:, 2:11].as_matrix()

mask = range(num_training, num_training + num_validation)
X_val = X_train[:,mask]
Y_val = Y_train[mask]

mask = range(num_training)
X_train = X_train[:,mask]
Y_train = Y_train[mask]

model = Sequential()
model.add(Dense(32, input_shape=(12288,), W_regularizer=l2(0.01)))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(output_dim=10))
model.add(Activation("linear"))

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train.T, Y_train, nb_epoch=5, batch_size=16)

score = model.evaluate(X_val.T, Y_val, batch_size=16)

print 
print score

model.save_weights('ava_simple.h5')


images = np.array
ava_dataset = "dataset/AVA/data"
root_path = os.getcwd()
ava_data_path = os.path.join(root_path, ava_dataset)

def image_to_pickle():
  count = 10000#len(os.listdir(ava_data_path))
  images = np.empty(count, dtype=object)
  i=0
  print "Loading Images..."
  for root, dirnames, filenames in os.walk(ava_data_path):
    for filename in sorted(filenames, key=lambda x: int(x.split('.')[0])):
      if i >= count:
        break
      if (i % 1000) == 0:
        print "Now processing " + str(i) + "/" + str(count)

      filepath = os.path.join(ava_data_path, filename)
      image = ndimage.imread(filepath, mode="RGB")
      image_resized = misc.imresize(image, (64, 64))
      images[i] = image_resized
      i=i+1
  pickle.dump(images, open("images.p", "wb"))

