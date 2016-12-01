from keras.layers import merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization

from keras import backend as K

from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D


from keras.utils.np_utils import to_categorical
from keras.callbacks import CSVLogger, ReduceLROnPlateau,ModelCheckpoint

from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D

from keras.optimizers import SGD,RMSprop
from keras.models import load_model


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

from scipy.misc import imread, imresize

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from googlenet_custom_layers import PoolHelper,LRN

from datetime import datetime

"""
Implementation of Inception Network v4 [Inception Network v4 Paper](http://arxiv.org/pdf/1602.07261v1.pdf) in Keras.
"""

if K.image_dim_ordering() == "th":
    channel_axis = 1
else:
    channel_axis = -1


class LRExponentialDecay(Callback):
    '''Reduce learning rate exponentially every n epochs.

    # Example
        ```python
            reduce_lr_exp = LRExponentialDecay(exponential_decay_factor=0.94, every_n_epochs=2)
            model.fit(X_train, Y_train, callbacks=[reduce_lr_exp])
        ```
    # Arguments
        exponential_decay_factor: hyperparameter of exponent by which learning rate will be reduced. 
        every_n_epochs: how often to reduce learning rate by exponential factor
    '''

    def __init__(self, exponential_decay_factor=0.94, every_n_epochs=2):
        super(Callback, self).__init__()
        self.exponential_decay_factor=exponential_decay_factor
        self.every_n_epochs=every_n_epochs
        self.wait=0
        self.reset()

    def reset(self):
        self.epoch_counter_curr=0

    def on_train_begin(self, logs={}):
        self.reset()

    def on_epoch_end(self, epoch, logs={}):
        if self.wait >= self.every_n_epochs:
            old_lr = K.get_value(self.model.optimizer.lr)
            new_lr = old_lr * np.exp(-self.exponential_decay_factor * (epoch / self.every_n_epochs))
            K.set_value(self.model.optimizer.lr, new_lr)
            self.wait = 0
        else:
            self.wait += 1

def inception_stem(input): # Input (299,299,3)
    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    c = Convolution2D(32, 3, 3, activation='relu', subsample=(2,2))(input)
    c = Convolution2D(32, 3, 3, activation='relu', )(c)
    c = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(c)

    c1 = MaxPooling2D((3,3), strides=(2,2))(c)
    c2 = Convolution2D(96, 3, 3, activation='relu', subsample=(2,2))(c)

    m = merge([c1, c2], mode='concat', concat_axis=channel_axis)

    c1 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(m)
    c1 = Convolution2D(96, 3, 3, activation='relu', )(c1)

    c2 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(m)
    c2 = Convolution2D(64, 7, 1, activation='relu', border_mode='same')(c2)
    c2 = Convolution2D(64, 1, 7, activation='relu', border_mode='same')(c2)
    c2 = Convolution2D(96, 3, 3, activation='relu', border_mode='valid')(c2)

    m2 = merge([c1, c2], mode='concat', concat_axis=channel_axis)

    p1 = MaxPooling2D((3,3), strides=(2,2), )(m2)
    p2 = Convolution2D(192, 3, 3, activation='relu', subsample=(2,2))(m2)

    m3 = merge([p1, p2], mode='concat', concat_axis=channel_axis)
    m3 = BatchNormalization(axis=1)(m3)
    m3 = Activation('relu')(m3)
    return m3


def inception_A(input):
    a1 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    a1 = Convolution2D(96, 1, 1, activation='relu', border_mode='same')(a1)

    a2 = Convolution2D(96, 1, 1, activation='relu', border_mode='same')(input)

    a3 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(input)
    a3 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(a3)

    a4 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(input)
    a4 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(a4)
    a4 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(a4)

    m = merge([a1, a2, a3, a4], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=1)(m)
    m = Activation('relu')(m)
    return m

def inception_B(input):
    b1 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    b1 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(b1)

    b2 = Convolution2D(384, 1, 1, activation='relu', border_mode='same')(input)

    b3 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    b3 = Convolution2D(224, 1, 7, activation='relu', border_mode='same')(b3)
    b3 = Convolution2D(256, 1, 7, activation='relu', border_mode='same')(b3)

    b4 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    b4 = Convolution2D(192, 1, 7, activation='relu', border_mode='same')(b4)
    b4 = Convolution2D(224, 7, 1, activation='relu', border_mode='same')(b4)
    b4 = Convolution2D(224, 1, 7, activation='relu', border_mode='same')(b4)
    b4 = Convolution2D(256, 7, 1, activation='relu', border_mode='same')(b4)

    m = merge([b1, b2, b3, b4], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=1)(m)
    m = Activation('relu')(m)
    return m

def inception_C(input):
    c1 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    c1 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(c1)

    c2 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)

    c3 = Convolution2D(384, 1, 1, activation='relu', border_mode='same')(input)
    c3_1 = Convolution2D(256, 1, 3, activation='relu', border_mode='same')(c3)
    c3_2 = Convolution2D(256, 3, 1, activation='relu', border_mode='same')(c3)

    c4 = Convolution2D(384, 1, 1, activation='relu', border_mode='same')(input)
    c4 = Convolution2D(192, 1, 3, activation='relu', border_mode='same')(c4)
    c4 = Convolution2D(224, 3, 1, activation='relu', border_mode='same')(c4)
    c4_1 = Convolution2D(256, 3, 1, activation='relu', border_mode='same')(c4)
    c4_2 = Convolution2D(256, 1, 3, activation='relu', border_mode='same')(c4)

    m = merge([c1, c2, c3_1, c3_2, c4_1, c4_2], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=1)(m)
    m = Activation('relu')(m)
    return m

def reduction_A(input, k=192, l=224, m=256, n=384):
    r1 = MaxPooling2D((3,3), strides=(2,2))(input)

    r2 = Convolution2D(n, 3, 3, activation='relu', subsample=(2,2))(input)

    r3 = Convolution2D(k, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(l, 3, 3, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(m, 3, 3, activation='relu', subsample=(2,2))(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=1)(m)
    m = Activation('relu')(m)
    return m

def reduction_B(input):
    r1 = input

    r2 = Convolution2D(192, 1, 1, activation='relu')(input)
    r2 = Convolution2D(192, 3, 3, activation='relu', border_mode='same')(r2)

    r3 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(256, 1, 7, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(320, 7, 1, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(320, 3, 3, activation='relu', border_mode='same')(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis)
    m_conv = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(m)
    m = BatchNormalization(axis=1)(m_conv)
    m = Activation('relu')(m)
    return m,m_conv


def create_inception_v4(input, heatmap=False):
    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x = inception_stem(input)

    # 4 x Inception A
    x = inception_A(x)
    x = inception_A(x)
    x = inception_A(x)
    x = inception_A(x)

    # Reduction A
    x = reduction_A(x)

    # 7 x Inception B
    x = inception_B(x)
    x = inception_B(x)
    x = inception_B(x)
    x = inception_B(x)
    x = inception_B(x)
    # x = inception_B(x)
    # x = inception_B(x)

    # Reduction B
    aesthetics, aesthetics_conv = reduction_B(x)
    semantics, semantics_conv = reduction_B(x)

    aesthetics = GlobalAveragePooling2D()(aesthetics)
    semantics = GlobalMaxPooling2D()(semantics)

    aesthetics = Dropout(0.8)(aesthetics)
    semantics = Dropout(0.8)(semantics)

    aesthetics = Dense(output_dim=2, activation='softmax',name="out_aesthetics")(aesthetics)
    semantics = Dense(output_dim=65, activation='softmax',name="out_semantics")(semantics)

    # # 3 x Inception C
    # x = inception_C(x)
    # x = inception_C(x)
    # x = inception_C(x)

    # # Average Pooling
    # x = AveragePooling2D((8,8))(x)

    # # Dropout
    # x = Dropout(0.8)(x)
    # x = Flatten()(x)

    # Output
    if heatmap:
        return aesthetics, aesthetics_conv, semantics, semantics_conv
    else:
        return aesthetics, semantics





delta = 0.0
store = HDFStore('../dataset_h5/labels.h5','r')
# delta = 1
ava_table = store['labels_train']

ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]
# X_train = np.hstack(X).reshape(10000,224,224,3)
# X = pickle.load( open("images_224.p", "rb"))
h5f = h5py.File('../dataset_h5/images_299x299_delta_{0}.h5'.format(delta),'r')
X_train = h5f['data_train']
#X_train = np.hstack(X).reshape(3,224,224,16160).T

#X_train = X_train.astype('float32')

Y_train = ava_table.ix[:, "good"].as_matrix()
Y_train = to_categorical(Y_train, 2)

Y_train_semantics = to_categorical(ava_table.ix[:,10:12].as_matrix())[:,1:]

X_test = h5f['data_test']
ava_test = store['labels_test']
Y_test = ava_test.ix[:, "good"].as_matrix()
Y_test = to_categorical(Y_test, 2)

Y_test_semantics = to_categorical(ava_test.ix[:,10:12].as_matrix())[:,1:]




ip = Input(shape=(3, 299, 299))

aesthetics, semantics = create_inception_v4(ip)
model = Model(input=ip, output=[aesthetics, semantics])

# out = model.predict(np.expand_dims(cv2.resize(cv2.imread(input_path).astype(np.float32),(299,299)).transpose((2,0,1)),axis=0))


rmsprop = RMSprop(lr=0.045, rho=0.9, epsilon=1e-08, decay=0.9)
# sgd = SGD(lr=0.001, decay=5e-4, momentum=0.9, nesterov=True)
model.compile(optimizer=rmsprop,loss='categorical_crossentropy', metrics=['accuracy'])

time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

checkpointer = ModelCheckpoint(filepath="googlenet_aesthetics_weights{}.h5".format(time_now), verbose=1, save_best_only=True)

reduce_lr_exp = LRExponentialDecay(exponential_decay_factor=0.94, every_n_epochs=2)

csv_logger = CSVLogger('training_gmp_aesthetics{}.log'.format(time_now))

model.fit(X_train,[Y_train,Y_train_semantics],nb_epoch=20, batch_size=31, shuffle="batch", validation_data=(X_test, [Y_test,Y_test_semantics]), callbacks=[csv_logger,checkpointer,reduce_lr_exp])#,class_weight = class_weight)
