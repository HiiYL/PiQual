from scipy import ndimage, misc
import cv2
import numpy as np
import os
import pickle
import pandas as pd
from pandas import HDFStore, DataFrame

import h5py

from keras.layers import Input, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import CSVLogger

from keras.optimizers import SGD

def VGG_19_functional(weights_path=None):

    inputs = Input(shape=(3, 224, 224))
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
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu',name='fc17')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu',name='fc18')(x)
    x = Dropout(0.5)(x)


    # m_x = Dense(4096, activation='relu',name='fc19')(x)
    # m_x = Dropout(0.5)(m_x)
    main_output = Dense(output_dim=1, name="main_output")(x)

    a_x = Dense(4096, activation='relu',name='fc20')(x)
    a_x = Dropout(0.5)(a_x)
    aux_output = Dense(20, activation='softmax', name='aux_output')(a_x)

    model = Model(input=inputs, output=[main_output,aux_output])

    if weights_path:
        model.load_weights(weights_path, by_name=True)


    return model

def VGG_19_functional_batchNorm(weights_path=None):

    inputs = Input(shape=(3, 224, 224))
    x = Convolution2D(64, 3, 3,border_mode='same',name='conv1_1')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64, 3, 3, activation='relu',name='conv1_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128, 3, 3, activation='relu',name='conv2_1')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128, 3, 3, activation='relu',name='conv2_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu',name='conv3_1')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu',name='conv3_2')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu',name='conv3_3')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu',name='conv3_4')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv4_1')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv4_2')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv4_3')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv4_4')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv5_1')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv5_2')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv5_3')(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv5_4')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu',name='fc17')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(4096, activation='relu',name='fc18')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)


    # m_x = Dense(4096, activation='relu',name='fc19')(x)
    # m_x = Dropout(0.5)(m_x)
    main_output = Dense(output_dim=1, name="main_output")(x)

    a_x = Dense(4096, activation='relu',name='fc20')(x)
    a_x = BatchNormalization()(a_x)
    a_x = Dropout(0.2)(a_x)
    aux_output = Dense(20, activation='softmax', name='aux_output')(a_x)

    model = Model(input=inputs, output=[main_output,aux_output])

    if weights_path:
        model.load_weights(weights_path, by_name=True)


    return model

delta = 0
store = HDFStore('../dataset_h5/labels.h5')

full_store = HDFStore('../dataset_h5/full_test_labels.h5')
# delta = 1
ava_table = store['labels']

ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]
# X_train = np.hstack(X).reshape(10000,224,224,3)
# X = pickle.load( open("images_224.p", "rb"))
h5f = h5py.File('../dataset_h5/images_224_delta_{0}.h5'.format(delta),'r')
X_train = h5f['data']
#X_train = np.hstack(X).reshape(3,224,224,16160).T

#X_train = X_train.astype('float32')

Y_train = ava_table.ix[:, "score"].as_matrix()


## Observation: Category 66 - Analog is never used in the entire dataset
## Can be tested with
## np.max(ava_table.ix[:,10:12])

Y_train_tag = to_categorical(ava_table.ix[:,10:12].as_matrix())[:,1:]


tag_occurence = Y_train_tag.sum(axis=0)
sorted_tags = sorted(range(len(tag_occurence)), key=lambda k: tag_occurence[k])
top_20_tags = sorted_tags[-20:]


Y_train_tag = Y_train_tag[:, top_20_tags]
# should produce [9, 23, 26, 1, 27, 39, 15, 28, 4, 37, 11, 19, 16, 0, 18, 21, 17, 13, 20, 14]
# offset by +1 when using due to removal of 0
# Y_train = to_categorical(Y_train, 2)



h5f_test = h5py.File('../dataset_h5/images_224.h5','r')
X_test = h5f_test['data_test']
ava_test = store['labels_test']
Y_test = ava_test.ix[:, "score"].as_matrix()

Y_test_tag = to_categorical(ava_test.ix[:,10:12].as_matrix())[:,1:]
Y_test_tag = Y_test_tag[:,top_20_tags]



model = VGG_19_functional('named_vgg19_weights.h5')

sgd = SGD(lr=1e-5, decay=5e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
    loss={'main_output': 'mse', 'aux_output': 'categorical_crossentropy'},
    loss_weights={'main_output': 1., 'aux_output': 0.05} # 1/M where M = 65
    )


csv_logger = CSVLogger('training_multitask_linear_0.05.log')

model.fit(X_train,
          {'main_output': Y_train, 'aux_output': Y_train_tag},
          nb_epoch=20, batch_size=32, shuffle="batch", validation_data=(X_test, [Y_test,Y_test_tag]), callbacks=[csv_logger])

model.evaluate(X_test, [Y_test, Y_test_tag])