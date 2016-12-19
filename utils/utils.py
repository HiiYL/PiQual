from shutil import copyfile
import pandas as pd
from pandas import HDFStore
import os
import h5py

import cv2
import numpy as np

def copyTestSet(dest_dir="test_images/"):
  ava_path = "dataset/AVA/data/"
  ava_data_path = os.path.join(os.getcwd(), ava_path)
  store = HDFStore('dataset_h5/labels.h5')
  test_labels = store['labels_test']
  imagesCount = test_labels.shape[0]

  i = 0
  for index, row in test_labels.iterrows():
    if (i % 1000) == 0:
      print("Now processing {} / {}".format(i,imagesCount))
    filename = str(index) + ".jpg"
    filepath = os.path.join(ava_data_path, filename)
    copyfile(filepath, dest_dir + filename)
    i=i+1


def computeAspectRatio():
  store = HDFStore('dataset_h5/labels.h5')
  ava_table = store['labels_train']


  ava_path = "dataset/AVA/data/"

  ava_data_path = os.path.join(os.getcwd(), ava_path)

  periodNum = ava_table.shape[0]

  aspect_ratio_list = np.empty(periodNum, dtype=np.float16)
  i = 0
  for index in ava_table.index:
    if (i % 1000) == 0:
      print('Now Processing {0}/{1}'.format(i,periodNum))
    filename = str(index) + ".jpg"
    filepath = os.path.join(ava_data_path, filename)
    height, width, channels = cv2.imread(filepath).shape
    aspect_ratio = float("{0:.2f}".format(height/width))
    aspect_ratio_list[i] = aspect_ratio
    i = i + 1

  ava_table[:, "aspect_ratio"] = aspect_ratio_list

  store['labels_train'] = ava_table
  stable_aspect = ava_table.ix[ (ava_table['aspect_ratio'] == 6) | (ava_table['aspect_ratio'] == 7) ]



  ava_test = store['labels_test']

  periodNum = ava_test.shape[0]

  aspect_ratio_list = np.empty(periodNum, dtype=np.float16)
  i = 0
  for index in ava_test.index:
    if (i % 1000) == 0:
      print('Now Processing {0}/{1}'.format(i,periodNum))
    filename = str(index) + ".jpg"
    filepath = os.path.join(ava_data_path, filename)
    height, width, channels = cv2.imread(filepath).shape
    aspect_ratio = float("{0:.2f}".format(height/width))
    aspect_ratio_list[i] = aspect_ratio
    i = i + 1
  ava_test[:, ""]


def stable_hdf5():
  channel = 3
  height = 224
  width = int((1/0.6) * height)

  delta = 0.0

  store = HDFStore('dataset_h5/labels.h5')
  ava_table = store['labels_train']

  filtered_ava_table = ava_table.ix[(ava_table['aspect_ratio'] == 6)]
  ava_path = "dataset/AVA/data/"
  ava_data_path = os.path.join(os.getcwd(), ava_path)
  h5f = h5py.File('dataset_h5/images_ar_6_delta_{}.h5'.format(delta),'w')

  # filtered_ava_table = ava_table.ix[ (ava_table['aspect_ratio'] == 6) | (ava_table['aspect_ratio'] == 7) ]

  periodNum = filtered_ava_table.shape[0]
  data = h5f.create_dataset("data_train",(periodNum,channel,height,width), dtype='uint8')

  print("Training Set: Converting AVA images to hdf5 with delta of {}...".format(delta))
  i=0
  for index in filtered_ava_table.index:
    if(i >= periodNum):
      break
    if (i % 1000) == 0:
      print('Now Processing {0}/{1}'.format(i,periodNum))
    filename = str(index) + ".jpg"
    filepath = os.path.join(ava_data_path, filename)
    im = cv2.resize(cv2.imread(filepath), (width, height)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    data[i] = im
    i=i+1