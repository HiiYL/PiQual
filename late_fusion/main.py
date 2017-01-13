from googlenet_custom_layers import PoolHelper,LRN
from keras.models import model_from_json
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
from keras.layers import Input, Embedding, LSTM, Dense, Activation, GRU,Convolution1D,Dropout
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D,GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.utils.np_utils import to_categorical
from keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


from datetime import datetime

from sklearn import svm, linear_model
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,roc_curve

from keras.models import load_model


X_train_text = np.load('out_train_text.npy')
X_train_image = np.load('out_train_images.npy')


X_test_text = np.load('X_test_text.npy')
X_test_image = np.load('X_test_images.npy')

import itertools

X_train = np.array([list(itertools.chain(m,n)) for m,n in zip(X_train_text,X_train_image)])

X_test = np.array([list(itertools.chain(m,n)) for m,n in zip(X_test_text,X_test_image)])

store = HDFStore('../dataset_h5/labels.h5','r')
ava_table = store['labels_train']
ava_test = store['labels_test']


Y_train = ava_table.ix[:, "good"].as_matrix()
# Y_train = to_categorical(Y_train, 2)

Y_test = ava_test.ix[:, "good"].as_matrix()
# Y_test = to_categorical(Y_test, 2)


def train(features, labels):
    X = features
    Y = labels

    clf = svm.LinearSVC()
    #clf = linear_model.SGDRegressor()
    clf.fit(X, Y)

    return clf


classifier = train(X_train, Y_train)
accuracy_score = accuracy_score(Y_test, classifier.predict(X_test))