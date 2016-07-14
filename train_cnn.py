from keras.models import Sequential
from scipy import ndimage, misc
import numpy as np
import os
import cPickle as pickle
import pandas as pd

from keras.layers import Dense, Activation
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

ava_table = pd.read_table("dataset/AVA/AVA.txt", delim_whitespace=True,
 index_col=1, header=None)
ava_table.sort_index(inplace=True)
X = pickle.load( open("images_224.p", "rb"))

num_training = 8000
num_test = 1000
X_train = np.hstack(X).reshape(10000,224,224,3)
Y_train = ava_table.ix[:, 3:10].as_matrix()

mask = range(num_training, num_training + num_test)
X_test = X_train[mask].transpose(1,2,3,0)
Y_test = Y_train[mask]

mask = range(num_training)
X_train = X_train[mask].transpose(1,2,3,0)
Y_train = Y_train[mask]

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train.T, Y_train, nb_epoch=5, batch_size=32,validation_split=0.1)
score = model.evaluate(X_test.T, Y_test, batch_size=32)

print 
print score


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
      image_resized = misc.imresize(image, (224, 224))
      images[i] = image_resized
      i=i+1
  pickle.dump(images, open("images_224.p", "wb"))
