from pandas import HDFStore
import pandas as pd


import os

import h5py
import cv2
import numpy as np
store = HDFStore('../dataset_h5/labels.h5')

ava_table = store['labels']

ava_path = "../dataset/AVA/data/"
style = pd.read_table('../dataset/AVA/style_image_lists/train.jpgl', index_col=0)
tag = pd.read_table('../dataset/AVA/style_image_lists/train.lab')

style.loc[:,'style'] = tag.as_matrix()

ava_with_style = style.join(ava_table, how='inner')

store['labels_with_style'] = ava_with_style


h5f = h5py.File('images_with_style.h5','w')

channel = 3
width = 224
height = 224

periodNum = ava_with_style.shape[0]

data = h5f.create_dataset("data_style_train",(periodNum,channel,width,height), dtype='uint8')


ava_path = "../dataset/AVA/data/"
ava_data_path = os.path.join(os.getcwd(), ava_path)

i=0
for index in ava_with_style.index:
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
	




style_test = pd.read_table('../dataset/AVA/style_image_lists/test.jpgl', index_col=0)
tag_test = pd.read_table('../dataset/AVA/style_image_lists/test.multilab')

style_test.loc[:,'style'] = tag_test.as_matrix()

ava_with_style_test = style_test.join(ava_table, how='inner')
store['labels_with_style_test'] = ava_with_style_test

data_test = h5f.create_dataset("data_style_test",(periodNum,channel,width,height), dtype='uint8')
i=0
for index in ava_with_style_test.index:
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
    data_test[i] = im 
  i=i+1

h5f.close()