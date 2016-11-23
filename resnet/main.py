from keras.applications.resnet50 import ResNet50
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.layers import Dense, Activation
from keras.models import Model

from keras.utils.np_utils import to_categorical
from keras.callbacks import CSVLogger, ReduceLROnPlateau,ModelCheckpoint

from keras.layers.pooling import GlobalAveragePooling2D

from keras.optimizers import SGD

import cv2

from scipy import ndimage, misc
import cv2
import numpy as np
import os
import pickle
import pandas as pd
from pandas import HDFStore, DataFrame

import h5py
import matplotlib.pyplot as plt

import sys
sys.setrecursionlimit(10000)

def process_image(image):
    im = np.copy(image)
    im = cv2.resize(im,(224,224))
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im

def deprocess_image(image):
    im = np.copy(image)
    im[:,:,0] += 103.939
    im[:,:,1] += 116.779
    im[:,:,2] += 123.

    im = im.transpose((1,2,0))

    return im



delta = 0.0
store = HDFStore('../dataset_h5/labels.h5','r')
# delta = 1
ava_table = store['labels_train']

ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]
# X_train = np.hstack(X).reshape(10000,224,224,3)
# X = pickle.load( open("images_224.p", "rb"))
h5f = h5py.File('../dataset_h5/images_224_delta_{0}.h5'.format(delta),'r')
X_train = h5f['data_train']
#X_train = np.hstack(X).reshape(3,224,224,16160).T

#X_train = X_train.astype('float32')

Y_train = ava_table.ix[:, "good"].as_matrix()
Y_train = to_categorical(Y_train, 2)

X_test = h5f['data_test']
ava_test = store['labels_test']
Y_test = ava_test.ix[:, "good"].as_matrix()
Y_test = to_categorical(Y_test, 2)



model = ResNet50(include_top=True, weights='imagenet', input_tensor=None)


## Pop the top off
model.layers.pop()
model.layers.pop()
model.layers.pop()

model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []

final_conv = Convolution2D(1024, 3, 3, activation='relu',name='conv6_1',border_mode = 'same')(model.outputs[0])

x = GlobalAveragePooling2D()(final_conv)
aesthetic_prediction_layer = Dense(output_dim=2, activation='softmax')(x)
model = Model(input=model.inputs, output=[aesthetic_prediction_layer,final_conv])


read_and_generate_heatmap("kitten.jpg","kitten-heatmap.jpg")

sgd = SGD(lr=0.001, decay=5e-4, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy', metrics=['accuracy'])





checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=1, min_lr=1e-6)

csv_logger = CSVLogger('training_gap_binary.log')

model.fit(X_train,Y_train,nb_epoch=20, batch_size=32, shuffle="batch", validation_data=(X_test, Y_test), callbacks=[csv_logger,checkpointer,reduce_lr])


def read_and_generate_heatmap(input_path, output_path):
    original_img = cv2.imread(input_path).astype(np.float32)

    width, height, _ = original_img.shape

    im = process_image(original_img)
    out = model.predict(im)

    class_weights = model.layers[-1].get_weights()[0]

    out[1] = out[1][0,:,:,:]

    #Create the class activation map.
    cam = np.zeros(dtype = np.float32, shape = out[1].shape[1:3])

    class_to_visualize = 1 # 0 for bad, 1 for good
    for i, w in enumerate(class_weights[:, class_to_visualize]):
            cam += w * out[1][i, :, :]
    print("predictions", out[0])
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    im = heatmap*0.5 + original_img
    cv2.imwrite(output_path, im)