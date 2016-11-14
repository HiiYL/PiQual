## Usage
#
# from image_to_hdf5 import ImageToHDF5

# ihdf = ImageToHDF5()
# ihdf.prepare_train()
# ihdf.prepare_test()

import pandas as pd
from pandas import HDFStore
import numpy as np
import os
import cv2

import h5py

class ImageToHDF5:

  def __init__(self,delta=1.0):
    self.store = HDFStore('dataset_h5/labels.h5')
    self.ava_table = self.store['labels_train']
    self.ava_path = "dataset/AVA/data/"
    self.ava_data_path = os.path.join(os.getcwd(), self.ava_path)
    self.h5f = h5py.File('dataset_h5/images_224_delta_{}.h5'.format(delta),'w')
    self.delta = delta

  def prepare_train(self, limit=None):

    channel = 3
    width= 224
    height = 224

    filtered_ava_table = self.ava_table[( abs(self.ava_table.score - 5) >= self.delta)]

    periodNum = limit if limit else filtered_ava_table.shape[0]
    data = self.h5f.create_dataset("data_train", (periodNum,channel,width,height), dtype='uint8')

    print("Training Set: Converting AVA images to hdf5 with delta of {}...".format(self.delta))
    i=0
    for index, row in filtered_ava_table.iterrows():
      if(i >= periodNum):
        break
      if (i % 1000) == 0:
        print('Now Processing {0}/{1}'.format(i,periodNum))
      filename = str(index) + ".jpg"
      filepath = os.path.join(self.ava_data_path, filename)
      im = cv2.resize(cv2.imread(filepath), (224, 224)).astype(np.float32)
      im[:,:,0] -= 103.939
      im[:,:,1] -= 116.779
      im[:,:,2] -= 123.68
      im = im.transpose((2,0,1))
      im = np.expand_dims(im, axis=0)
      data[i] = im
      i=i+1

  def close(self):
    self.store.close()
    self.h5f.close()

  def prepare_test(self):

    channel = 3
    width= 224
    height = 224

    print("Test Set: Converting AVA images to hdf5...")
    test_labels = self.store['labels_test']
    imagesCount = test_labels.shape[0]

    data = self.h5f.create_dataset("data_test", (imagesCount,channel,width,height), dtype='uint8')
    i=0
    for index, row in test_labels.iterrows():
      if (i % 1000) == 0:
        print("Now processing {} / {}".format(i,imagesCount))
      filename = str(index) + ".jpg"
      filepath = os.path.join(self.ava_data_path, filename)
      im = cv2.resize(cv2.imread(filepath), (224, 224)).astype(np.float32)
      im[:,:,0] -= 103.939
      im[:,:,1] -= 116.779
      im[:,:,2] -= 123.68
      im = im.transpose((2,0,1))
      im = np.expand_dims(im, axis=0)
      data[i] = im
      i=i+1

if __name__ == "__main__":
  ihdf = ImageToHDF5(delta=0.0)
  ihdf.prepare_train()
  ihdf.prepare_test()
  ihdf.close()
