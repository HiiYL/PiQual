from scipy import ndimage, misc
import cv2
import numpy as np
import os
import cPickle as pickle
import pandas as pd
from pandas import HDFStore, DataFrame

import h5py

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD

num_training = 500
num_test = 10000

store = HDFStore('labels.h5')
# delta = 1
ava_table = store['labels']
# X_train = np.hstack(X).reshape(10000,224,224,3)
# X = pickle.load( open("images_224.p", "rb"))
h5f = h5py.File('images_224_50.h5','r')
# X = h5f['images'][:]
X_train = h5f['data']

# X_train = X_train.astype('float32')

Y_train = ava_table.ix[:, "good"].as_matrix()
Y_train = to_categorical(Y_train, 2)

mask = range(num_training, num_training + num_test)
X_test = X_train[mask]
Y_test = Y_train[mask]

mask = range(num_training)
X_train = X_train[mask]
Y_train = Y_train[mask]

X_mean = np.mean(X_train)
X_train -= X_mean
X_train /= 255

X_test -= np.mean(X_test)
X_test /= 255

weights_path = os.path.join(os.getcwd(), "vgg16_weights.h5")

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
model.add(Convolution2D(64, 3, 3, activation='relu',trainable=False))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu',trainable=False))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu',trainable=False))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu',trainable=False))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu',trainable=False))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu',trainable=False))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu',trainable=False))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu',trainable=False))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu',trainable=False))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu',trainable=False))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu',trainable=False))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu',trainable=False))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu',trainable=False))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))

model.load_weights(weights_path)

model.layers.pop()
model.layers.pop()
model.add(Dropout(0.5))
model.add(Dense(output_dim=2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train,validation_split=0.1, nb_epoch=10)


model.save_weights('ava_vgg.h5')

score = model.evaluate(X_test.T, Y_test)
print 
print('Test score:', score[0])
print('Test accuracy:', score[1])
print
print "Predictions"


print model.predict(np.expand_dims(X_test.T[0], axis=0))
print model.predict(np.expand_dims(X_test.T[1], axis=0))
print model.predict(np.expand_dims(X_test.T[2], axis=0))
print model.predict(np.expand_dims(X_test.T[3], axis=0))
print model.predict(np.expand_dims(X_test.T[4], axis=0))

filepath = os.path.join(os.getcwd(), "forest.jpg")
image = ndimage.imread(filepath, mode="RGB")
image_resized = misc.imresize(image, (224, 224)).astype('float32')
image_resized -= X_mean
image_resized /= 255
print model.predict(np.expand_dims(image_resized.T, axis=0))

filepath = os.path.join(os.getcwd(), "test.jpg")
image = ndimage.imread(filepath, mode="RGB")
image_resized = misc.imresize(image, (224, 224)).astype('float32')
image_resized -= X_mean
image_resized /= 255
print model.predict(np.expand_dims(image_resized.T, axis=0))




def image_to_pickle():
  ava_path = "dataset/AVA/data/"
  ava_data_path = os.path.join(os.getcwd(), ava_path)
  periodNum = 500 #len(os.listdir(ava_data_path))
  channel = 3
  width= 224
  height = 224
  store = HDFStore('labels.h5')
  filtered_ava = store['labels']

  h5f = h5py.File('images_224_50.h5', 'w')

  data = h5f.create_dataset("data", (periodNum,channel,width,height), dtype='uint8')

  print "Loading Images..."

  i=0
  invalid_indices = []
  for index, row in filtered_ava.iterrows():
    if (i >= periodNum):
      break
    if (i % 1000) == 0:
      print "Now processing " + str(i) + "/" + str(periodNum)
    filename = str(index) + ".jpg"
    filepath = os.path.join(ava_data_path, filename)
    try:
      image = ndimage.imread(filepath, mode="RGB")
      image_resized = misc.imresize(image, (224, 224)).T
      data[i] = np.expand_dims(image_resized,axis=0)
      i=i+1
    except IOError:
      invalid_indices.append(index)
      print filename + " at position " + str(i) + "is missing or invalid."

    if invalid_indices:
      try:
        filtered_ava = filtered_ava.drop(invalid_indices)
        del store['labels']
        store['labels'] = filtered_ava
      except ValueError:
        print "UHOH THIS SHOULDNT HAVE HAPPENED IMAGE TO PICKLE"


  h5f.close()
  store.close()
  # filtered_ava.save_pickle('filtered_ava.p')

