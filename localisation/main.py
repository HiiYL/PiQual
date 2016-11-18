from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Input, Dense, Activation
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.models import Model
from keras import backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import CSVLogger

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

def VGG_19_GAP_functional(weights_path=None,heatmap=False):

    inputs = Input(shape=(3, None, None))

    x = Convolution2D(64, 3, 3, activation='relu',border_mode='same',name='conv1_1')(inputs)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64, 3, 3, activation='relu',name='conv1_2')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128, 3, 3, activation='relu',name='conv2_1')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128, 3, 3, activation='relu',name='conv2_2')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu',name='conv3_1')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu',name='conv3_2')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu',name='conv3_3')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu',name='conv3_4')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv4_1')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv4_2')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv4_3')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv4_4')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv5_1')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv5_2')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv5_3')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv5_4')(x)

    x = ZeroPadding2D((1,1))(x)
    final_conv = Convolution2D(1024, 3, 3, activation='relu',name='conv6_1')(x)

    x = GlobalAveragePooling2D()(final_conv)

    main_output = Dense(2, activation = 'softmax', name="main_output")(x)
    aux_output = final_conv


    if heatmap:
        model = Model(input=inputs, output=[main_output,aux_output])
    else:
        model = Model(input=inputs, output=main_output)#[main_output,aux_output])

    if weights_path:
        model.load_weights(weights_path,by_name=True)

    return model

def process_image(image):
    im = np.copy(original_img)
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


if __name__ == "__main__":



    sgd = SGD(lr=0.001, decay=5e-4, momentum=0.9, nesterov=True)
    model = VGG_19_GAP_functional(weights_path='aesthestic_gap_weights_1.h5',heatmap=False)

    # model.compile(optimizer=sgd, loss='mse')


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


    model.compile(optimizer=sgd,loss='categorical_crossentropy', metrics=['accuracy'])




    csv_logger = CSVLogger('training_gap_binary.log')

    #model.fit(X_train,Y_train,
    #    nb_epoch=20, batch_size=32, shuffle="batch", validation_data=(X_test, Y_test), callbacks=[csv_logger])


    # image_path = "highasfuck.png"

    # original_img = cv2.imread(image_path).astype(np.float32)

    # width, height, _ = original_img.shape

    # im = process_image(original_img)

    results = model.evaluate(X_test,Y_test)




    # 

    ava_path = "../dataset/AVA/data/"

    for index in ava_test.iloc[:25].index:
        image_name = str(index) + ".jpg"
        original_img = cv2.imread(ava_path + image_name).astype(np.float32)

        width, height, _ = original_img.shape

        im = process_image(original_img)
        out = model.predict(im)
        output_path = "output/" + image_name

        
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
        heatmap[np.where(cam < 0.4)] = 0
        im = heatmap*0.5 + original_img
        cv2.imwrite(output_path, im)