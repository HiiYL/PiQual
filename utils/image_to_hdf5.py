## Usage
#
# from image_to_hdf5 import ImageToHDF5

# ihdf = ImageToHDF5()
# ihdf.prepare_train_no_resize()
# ihdf.prepare_test()

import pandas as pd
from pandas import HDFStore
import numpy as np
import os
import cv2

import h5py
# from fuel.datasets import H5PYDataset

class ImageToHDF5:

  def __init__(self,delta=1.0,resize=True):
    self.store = HDFStore('dataset_h5/labels.h5')
    self.ava_table = self.store['labels_train']
    self.ava_path = "dataset/AVA/data/"
    self.ava_data_path = os.path.join(os.getcwd(), self.ava_path)
    self.h5f = h5py.File('dataset_h5/images_299x299_delta_{}.h5'.format(delta),'w')
    self.delta = delta

  def prepare_train(self, limit=None, dim=(224,224)):
    channel = 3
    width, height = dim
    filtered_ava_table = self.ava_table[( abs(self.ava_table.score - 5) >= self.delta)]

    periodNum = limit if limit else filtered_ava_table.shape[0]
    data = self.h5f.create_dataset("data_train",(periodNum,channel,width,height), dtype='uint8')

    print("Training Set: Converting AVA images to hdf5 with delta of {}...".format(self.delta))
    i=0
    for index in filtered_ava_table.index:
      if(i >= periodNum):
        break
      if (i % 1000) == 0:
        print('Now Processing {0}/{1}'.format(i,periodNum))
      filename = str(index) + ".jpg"
      filepath = os.path.join(self.ava_data_path, filename)
      im = cv2.resize(cv2.imread(filepath), (width, height)).astype(np.float32)
      im[:,:,0] -= 103.939
      im[:,:,1] -= 116.779
      im[:,:,2] -= 123.68
      im = im.transpose((2,0,1))
      im = np.expand_dims(im, axis=0)
      data[i] = im
      i=i+1

  def prepare_test(self,dim=(224,224)):

    channel = 3
    width, height = dim

    print("Test Set: Converting AVA images to hdf5...")
    test_labels = self.store['labels_test']
    imagesCount = test_labels.shape[0]

    data = self.h5f.create_dataset("data_test", (imagesCount,channel,width,height), dtype='uint8')
    i=0
    for index in test_labels.index:
      if (i % 1000) == 0:
        print("Now processing {} / {}".format(i,imagesCount))
      filename = str(index) + ".jpg"
      filepath = os.path.join(self.ava_data_path, filename)
      im = cv2.resize(cv2.imread(filepath), (width, height)).astype(np.float32)
      im[:,:,0] -= 103.939
      im[:,:,1] -= 116.779
      im[:,:,2] -= 123.68
      im = im.transpose((2,0,1))
      im = np.expand_dims(im, axis=0)
      data[i] = im
      i=i+1

  def close(self):
    self.store.close()
    self.h5f.flush()
    self.h5f.close()

  # def prepare_train_no_resize(self, limit=None):
  #   channel = 3
  #   width= 224
  #   height = 224

  #   filtered_ava_table = self.ava_table[( abs(self.ava_table.score - 5) >= self.delta)]

  #   periodNum = 5 #limit if limit else filtered_ava_table.shape[0]


  #   dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
  #   data = self.h5f.create_dataset('data_train', (periodNum,), dtype=dtype)
  #   data.dims[0].label = 'batch'

  #   data_shapes = self.h5f.create_dataset('shapes', (periodNum, 3), dtype='int32')

  #   print("Training Set: Converting AVA images to hdf5 with delta of {}...".format(self.delta))
  #   i=0
  #   for index in filtered_ava_table.index:
  #     if(i >= periodNum):
  #       break
  #     if (i % 1000) == 0:
  #       print('Now Processing {0}/{1}'.format(i,periodNum))
  #     filename = str(index) + ".jpg"
  #     filepath = os.path.join(self.ava_data_path, filename)
  #     im = cv2.imread(filepath).astype(np.float32)
  #     im[:,:,0] -= 103.939
  #     im[:,:,1] -= 116.779
  #     im[:,:,2] -= 123.68
  #     im = im.transpose((2,0,1))
  #     # im = np.expand_dims(im, axis=0)

  #     print(im.shape)
  #     data[i] = im.flatten()
  #     data_shapes[i] = im.shape
  #     i=i+1

  #   data.dims.create_scale(data_shapes, 'shapes')
  #   data.dims[0].attach_scale(data_shapes)

  #   data_shape_labels = self.h5f.create_dataset('shape_labels', (3,), dtype='S7')
  #   data_shape_labels[...] = ['channel'.encode('utf8'), 'height'.encode('utf8'),'width'.encode('utf8')]
  #   data.dims.create_scale(data_shape_labels, 'shape_labels')
  #   data.dims[0].attach_scale(data_shape_labels)



  #   split_dict = {
  #    'train': {'data_train': (1,4)},
  #    'test': {'data_train': (4, 5)}}

  #   self.h5f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

if __name__ == "__main__":
  ihdf = ImageToHDF5(delta=1.0)
  ihdf.prepare_train()
  ihdf.prepare_test()
  ihdf.close()



def retrieval():
  import h5py                                                             
  from fuel.datasets import H5PYDataset

  train_set  = H5PYDataset('dataset_h5/images_no_resize_delta_0.0.h5',which_sets=['train'], sources=('data_train',)) 
  print(train_set.axis_labels['data_train'])
  handle = train_set.open()
  data_no_resize,  = train_set.get_data(handle, slice(0,3))

  sgd = SGD(lr=0.001, decay=5e-4, momentum=0.9, nesterov=True)
  model = VGG_19_GAP_functional(weights_path='aesthestic_gap_weights_1.h5',heatmap=True)

  # model.compile(optimizer=sgd, loss='mse')


  delta = 0.0
  store = HDFStore('dataset_h5/labels.h5','r')
  # delta = 1
  ava_table = store['labels_train'][:3]

  ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]
  # X_train = np.hstack(X).reshape(10000,224,224,3)
  # X = pickle.load( open("images_224.p", "rb"))
  h5f = h5py.File('dataset_h5/images_224_delta_{0}.h5'.format(delta),'r')

  h5f2 = h5py.File('dataset_h5/images_no_resize_delta_0.0.h5'.format(delta),'r')
  X_train = h5f['data_train']
  #X_train = np.hstack(X).reshape(3,224,224,16160).T

  #X_train = X_train.astype('float32')

  Y_train = ava_table.ix[:, "good"].as_matrix()
  Y_train = to_categorical(Y_train, 2)

  X_test = h5f['data_test']
  ava_test = store['labels_test']
  Y_test = ava_test.ix[:, "good"].as_matrix()
  Y_test = to_categorical(Y_test, 2)


  model = VGG_19_GAP_functional(weights_path='localisation/aesthestic_gap_weights_1.h5',heatmap=False)

  sgd = SGD(lr=0.001, decay=5e-4, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd,loss='categorical_crossentropy', metrics=['accuracy'])

  model.fit(X_train,Y_train,nb_epoch=20, batch_size=32, shuffle='batch')

