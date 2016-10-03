from scipy import ndimage, misc
import cv2
import numpy as np
import os
import pickle
import pandas as pd
from pandas import HDFStore, DataFrame

import h5py

from keras.models import Sequential, load_model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Activation
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD


def VGG_19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def good(c):
  if c >= 5:
    return 1
  else:
    return 0
if __name__ == "__main__":

    # num_training = 9000
    # num_test = 1000

    delta = 0
    store = HDFStore('dataset_h5/labels.h5')

    full_store = HDFStore('dataset_h5/full_test_labels.h5')
    # delta = 1
    ava_table = store['labels']

    ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]
    # X_train = np.hstack(X).reshape(10000,224,224,3)
    # X = pickle.load( open("images_224.p", "rb"))
    h5f = h5py.File('dataset_h5/images_224_delta_{0}.h5'.format(delta),'r')
    X_train = h5f['data']
    #X_train = np.hstack(X).reshape(3,224,224,16160).T

    #X_train = X_train.astype('float32')

    Y_train = ava_table.ix[:, "score"].as_matrix()
    # Y_train = to_categorical(Y_train, 2)



    h5f_test = h5py.File('dataset_h5/images_224.h5','r')
    X_test = h5f_test['data_test']
    ava_test = store['labels_test']
    Y_test = ava_test.ix[:, "score"].as_matrix()
    # Y_test = to_categorical(Y_test, 2)



    model = VGG_19('vgg19_weights.h5')
    #for layer in model.layers[:19]:
    #    layer.trainable = False
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(output_dim=1))

    #model.save_weights('ava_vgg_19_{0}.h5'.format(delta))

    sgd = SGD(lr=1e-5, decay=5e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                          patience=0, min_lr=0, verbose=1)
    hist = model.fit(X_train, Y_train, nb_epoch=20,shuffle="batch", callbacks=[reduce_lr])


    prediction = model.predict(X_test)

    binary_pred = np.apply_along_axis(good, 1,prediction)


    Y_test_binary = ava_test.ix[:,"good"].as_matrix()

    prediction_accuracy = np.mean(binary_pred == Y_test_binary[:,1])


    model.save_weights('linear_ava_{0}.h5'.format(delta))

    score = model.evaluate(X_test, Y_test)
    print()
    print('Test score:{0}'.format(score[0]))
    print('Test accuracy:{0}'.format(score[1]))
    print()
    print("Predictions")