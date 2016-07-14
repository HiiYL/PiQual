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
ava_table = pd.read_pickle('filtered_ava.p')
X = pickle.load( open("images.p", "rb"))

num_training = 9000
num_test = 1000
X_train = np.hstack(X).reshape(10000,-1).T #.transpose(0,2,3,1).astype("float")
Y_train = ava_table.ix[:, "score"].as_matrix()

mask = range(num_training, num_training + num_test)
X_test = X_train[:,mask]
Y_test = Y_train[mask]

mask = range(num_training)
X_train = X_train[:,mask]
Y_train = Y_train[mask]

model = Sequential()
model.add(Dense(32, input_shape=(12288,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(output_dim=1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train.T, Y_train, nb_epoch=32, validation_split=0.1)

score = model.evaluate(X_test.T, Y_test)

print 
print score

model.save_weights('ava_simple.h5')

def image_to_pickle():
  ava_path = "dataset/AVA/data/"
  ava_data_path = os.path.join(os.getcwd(), ava_path)
  count = 10000#len(os.listdir(ava_data_path))
  filtered_ava = pd.read_pickle('filtered_ava.p')

  images = np.empty(count, dtype=object)
  i=0
  print "Loading Images..."
  # for root, dirnames, filenames in os.walk(ava_data_path):
  #   for filename in sorted(filenames, key=lambda x: int(x.split('.')[0])):
  #     if i >= count:
  #       break
  #     if (i % 1000) == 0:
  #       print "Now processing " + str(i) + "/" + str(count)
  #     image_index = int(filename.split(".jpg")[0])
  #     try:
  #       filtered_ava.iloc[[image_index]]
  #       filepath = os.path.join(ava_data_path, filename)
  #       image = ndimage.imread(filepath, mode="RGB")
  #       image_resized = misc.imresize(image, (64, 64))
  #       images[i] = image_resized
  #     except KeyError:
  #       print filename + " is missing or invalid."
  #     i=i+1
  i=0
  count=10000
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
    except IOError:
      invalid_indices.append(index)
      print filename + " at position " + str(i) + "is missing or invalid."
    i=i+1
  filtered_ava.drop(invalid_indices)
  filtered_ava.save_pickle('filtered_ava.p')
  pickle.dump(images, open("images.p", "wb"))


  # ava_images = pd.Series(images, index=filtered_ava.head(count).index)
  # filtered_ava['image'] = ava_images
  # filtered_ava.to_pickle('ava_dataset.p')
  # pickle.dump(images, open("ava_dataset.p", "wb"))


def preprocess():
  ava_table = pd.read_table("dataset/AVA/AVA.txt", delim_whitespace=True,
 index_col=1, header=None)
  ava_table.sort_index(inplace=True)

  averaged_score = pd.Series(index=ava_table.index)

  for index, row in ava_table.iterrows():
    sum_ratings = row[2:11].sum()
    count=1
    sum_ratings_weighted = 0
    for column in row[2:11]:
      sum_ratings_weighted += column * count
      count=count+1
    averaged_score[index] = sum_ratings_weighted / sum_ratings

  ava_table['score'] = averaged_score
  ava_table.to_pickle("ava.p")

def validate_table_against_images():
 
  filtered_ava = pd.read_pickle('filtered_ava.p')
  for index, row in filtered_ava.iterrows():
    filepath = os.path.join(os.getcwd(), "test.jpg")



