from scipy import ndimage, misc
import numpy as np
import os
import cPickle as pickle
import pandas as pd
from pandas import HDFStore, DataFrame
import h5py
from __future__ import division

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2, activity_l2
from keras.utils.np_utils import to_categorical
from keras.utils.io_utils import HDF5Matrix
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD

# ava_table = pd.read_table("dataset/AVA/AVA.txt", delim_whitespace=True)
store = HDFStore('labels.h5')
# delta = 1
ava_table = store['labels']
# ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]
X = pickle.load( open("images.p", "rb"))

num_training = 9000
num_test = 1000
X_train = np.hstack(X).reshape(10000,-1).T #.transpose(0,2,3,1).astype("float")
Y_train = ava_table.ix[:, "good"].as_matrix()
Y_train = to_categorical(Y_train, 2)

mask = range(num_training, num_training + num_test)
X_test = X_train[:,mask]
Y_test = Y_train[mask]

mask = range(num_training)
X_train = X_train[:,mask]
Y_train = Y_train[mask]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

X_train /= 255
X_test /= 255

model = Sequential()
model.add(Dense(64, input_dim=12288, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train.T, Y_train, nb_epoch=100, batch_size=32, validation_split=0.1)

score = model.evaluate(X_test.T, Y_test)

print 
print('Test score:', score[0])
print('Test accuracy:', score[1])


print model.predict(np.expand_dims(X_test.T[0], axis=0))
print model.predict(np.expand_dims(X_test.T[1], axis=0))
print model.predict(np.expand_dims(X_test.T[2], axis=0))
print model.predict(np.expand_dims(X_test.T[3], axis=0))
print model.predict(np.expand_dims(X_test.T[4], axis=0))

filepath = os.path.join(os.getcwd(), "forest.jpg")
image = ndimage.imread(filepath, mode="RGB")
image_resized = misc.imresize(image, (64, 64))
print np.argmax(model.predict(image_resized.reshape(1,12288)))

filepath = os.path.join(os.getcwd(), "test.jpg")
image = ndimage.imread(filepath, mode="RGB")
image_resized = misc.imresize(image, (64, 64))
print np.argmax(model.predict(image_resized.reshape(1,12288)))

model.save_weights('ava_simple.h5')

store.close()


def image_to_pickle():
  delta=0
  store = HDFStore('labels.h5')

  ava_path = "dataset/AVA/data/"
  ava_data_path = os.path.join(os.getcwd(), ava_path)
  filtered_ava = store['labels']

  #filtered_ava = filtered_ava[( abs(filtered_ava.score - 5) >= delta)]
  

  count = 10000

  images = np.empty(count, dtype=object)
  print "Loading Images..."
  i=0
  invalid_indices = []
  for index, row in filtered_ava.iterrows():
    if i >= count:
      break
    if (i % 1000) == 0:
      print "Now processing " + str(i) + "/" + str(count)
    filename = str(index) + ".jpg"
    filepath = os.path.join(ava_data_path, filename)
    try:
      image = ndimage.imread(filepath, mode="RGB")
      image_resized = misc.imresize(image, (64, 64))
      images[i] = image_resized
      i=i+1
    except IOError:
      invalid_indices.append(index)
      print filename + " at position " + str(i) + " is missing or invalid."

  if invalid_indices:
    try:
      filtered_ava = filtered_ava.drop(invalid_indices)
      del store['labels']
      store['labels'] = filtered_ava
    except ValueError:
      print "UHOH THIS SHOULDNT HAVE HAPPENED IMAGE TO PICKLE"

  store.close()

  h5f = h5py.File('images.h5', 'w')
  h5f.create_dataset('images', data=images.tolist())
  h5f.close()

  # filtered_ava.drop(invalid_indices)
  # filtered_ava.save_pickle('filtered_ava.p')


def good(c):
  if c['score'] >= 5:
    return 1
  else:
    return 0
def preprocess():
  ava_table = pd.read_table("dataset/AVA/AVA.txt", delim_whitespace=True, index_col=0,
   header=None,usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])#,names=["Image_ID",1,2,3,4,5,6,7,8,9,10,"ST_ID", "ST_ID", "C_ID"])
  ava_table.sort_index(inplace=True)

  ava_score_only = pd.DataFrame(index=ava_table.index)

  averaged_score = pd.Series(index=ava_table.index)

  print "Preprocessing..."
  for index, row in ava_table.iterrows():
    sum_ratings = row[1:11].sum()
    count=1
    sum_ratings_weighted = 0
    for column in row[1:11]:
      sum_ratings_weighted += column * count
      count=count+1
    averaged_score[index] = sum_ratings_weighted / sum_ratings

  # ava_table['score'] = averaged_score
  ava_score_only['score'] = averaged_score

  ava_score_only['good'] = ava_score_only.apply(good, axis=1)

  store = HDFStore('labels.h5')

  store['labels'] = ava_score_only

  store.close()

  # ava_table.to_pickle("ava.p")


def validate_table_against_images():
 
  filtered_ava = pd.read_pickle('filtered_ava.p')
  for index, row in filtered_ava.iterrows():
    filepath = os.path.join(os.getcwd(), "test.jpg")



def good(c):
  if c['score'] >= 5:
    return 1
  else:
    return 0