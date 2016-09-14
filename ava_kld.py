from scipy import ndimage, misc
import cv2
import numpy as np
import os
import pickle
import pandas as pd
from pandas import HDFStore, DataFrame

from keras.models import load_model

from keras import backend as K

import h5py

from keras.models import Sequential
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

def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 50)
    y_pred = K.clip(y_pred, K.epsilon(), 50)

    begin = K.log(y_true[:,1] / y_pred[:,1])

    middle = ( ( ( K.square(y_pred[:,1]) ) + ( K.square( y_pred[:,0] - y_true[:,0] ) ) ) /  (2 * K.square(y_true[:,0]) ))

    return K.sum( begin + middle - 1/2, axis=-1)

if __name__ == "__main__":

    # num_training = 9000
    # num_test = 1000

    delta = 1.0
    store = HDFStore('dataset_h5/labels.h5')
    # delta = 1
    ava_table = store['labels_train']

    ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]
    # X_train = np.hstack(X).reshape(10000,224,224,3)
    # X = pickle.load( open("images_224.p", "rb"))
    h5f = h5py.File('dataset_h5/images_224_delta_{0}.h5'.format(delta),'r')
    X_train = h5f['data']
    #X_train = np.hstack(X).reshape(3,224,224,16160).T

    #X_train = X_train.astype('float32')

    # Y_train = ava_table.ix[:, "good"].as_matrix()
    # Y_train = to_categorical(Y_train, 2)

    Y_train = ava_table.ix[:,['score','standard_deviation']].as_matrix()

    Y_train /= np.max(Y_train,axis=0)

    #Y_train = ava_table.ix[:,['score']].as_matrix()


    h5f_test = h5py.File('dataset_h5/images_224.h5','r')
    X_test = h5f_test['data_test']
    ava_test = store['labels_test']

    Y_test = ava_test.ix[:,['score','standard_deviation']].as_matrix()

    Y_test /= np.max(Y_train, axis=0)

    #Y_test[:,0] /= 10
    #Y_test[:,1] -= 1

   # Y_test = ava_test.ix[:,['score']].as_matrix()
    # Y_test = ava_test.ix[:, "good"].as_matrix()
    # Y_test = to_categorical(Y_test, 2)

    # mask = range(num_training, num_training + num_test)
    # X_test = X_train[mask].transpose(1,2,3,0)
    # Y_test = Y_train[mask]

    # mask = range(num_training)
    # X_train = X_train[mask].transpose(1,2,3,0)
    # Y_train = Y_train[mask]

    # X_mean = np.mean(X_train)
    # X_train -= X_mean
    # X_train /= 255

    # X_test -= np.mean(X_test)
    # X_test /= 255

    weights_path = os.path.join(os.getcwd(), "ava_vgg_19_{0}_5.h5".format(delta))



    model = VGG_19('vgg19_weights.h5')
    #for layer in model.layers[:19]:
    #    layer.trainable = False
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(output_dim=2, activation='sigmoid'))

    #model.save_weights('ava_vgg_19_{0}.h5'.format(delta))

    sgd = SGD(lr=0.001, decay=5e-4, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)
    model.fit(X_train, Y_train, nb_epoch=10,shuffle="batch")


    model.save_weights(('ava_vgg_19_{0}.h5'.format(delta))

    score = model.evaluate(X_test, Y_test)
    print()
    print('Test score:{0}'.format(score[0]))
    print('Test accuracy:{0}'.format(score[1]))
    print()
    print("Predictions")