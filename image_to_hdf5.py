## Usage
#
#  from image_to_hdf5 import ImageToHDF5
#
#  ihdf = ImageToHDF5()
#  ihdf.prepare_train()
#  ihdf.prepare_test()

import pandas as pd
from pandas import HDFStore
import numpy as np
import os

class ImageToHDF5:

  def __init__(self):
    self.store = HDFStore('dataset_h5/labels.h5')
    self.ava_table = store['labels_train']
    self.ava_path = "dataset/AVA/data/"
    self.ava_data_path = os.path.join(os.getcwd(), self.ava_path)

  def test(self):
    print("Hello There")

  def prepare_train(self,delta=1.0, limit=None):

    filtered_ava_table = self.ava_table[( abs(self.ava_table.score - 5) >= delta)]

    channel = 3
    width= 224
    height = 224

    h5f = h5py.File('dataset_h5/images_224_delta_1.5.h5', 'w')

    periodNum = limit if limit else filtered_ava_table.shape[0]
    data = h5f.create_dataset("data", (periodNum,channel,width,height), dtype='uint8')

    print("Training Set: Converting AVA images to hdf5...")
    i=0
    invalid_indices = []
    for index, row in filtered_ava_table.iterrows():
      if(i >= periodNum):
        break
      if (i % 1000) == 0:
        print('Now Processing {0}/{1}'.format(i,periodNum))
      filename = str(index) + ".jpg"
      filepath = os.path.join(self.ava_data_path, filename)
      image = ndimage.imread(filepath, mode="RGB")
      image_resized = misc.imresize(image, (224, 224)).T
      data[i] = np.expand_dims(image_resized,axis=0)
      i=i+1
    h5f.close()
    store.close()

  def prepare_test(self,delta=0.0):
    print("Test Set: Converting AVA images to hdf5...")
    test_labels = store['labels_test']
    imagesCount = test_images.shape[0]
    data = h5f.create_dataset("data_test", (imagesCount,channel,width,height), dtype='uint8')
    i=0
    for index, row in test_labels.iterrows():
      if (i % 1000) == 0:
        print("Now processing {} / {}").format(i,imagesCount)
      filename = str(index) + ".jpg"
      filepath = os.path.join(self.ava_data_path, filename)
      image = ndimage.imread(filepath, mode="RGB")
      image_resized = misc.imresize(image, (width, height)).T
      data[i] = np.expand_dims(image_resized,axis=0)
      i=i+1

if __name__ == "__main__":
  ihdf = ImageToHDF5()
  ihdf.prepare_train()
  ihdf.prepare_test()
