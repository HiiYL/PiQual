from scipy import ndimage, misc
import cv2
import numpy as np
import os
import pickle
import pandas as pd
from pandas import HDFStore, DataFrame

import h5py

from sklearn import svm, linear_model
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,roc_curve

from keras.models import load_model

delta = 1.0
store = HDFStore('dataset_h5/labels.h5')


ava_table = store['labels_train']
ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]


h5f = h5py.File('dataset_h5/images_224_delta_{0}.h5'.format(delta),'r')
X_train = h5f['data']
Y_train = ava_table.ix[:, "good"].as_matrix()
# Y_train = to_categorical(Y_train, 2)

h5f_test = h5py.File('dataset_h5/images_224.h5','r')
X_test = h5f_test['data_test']
Y_test = store['labels_test'].ix[:, "good"].as_matrix()
# Y_test = to_categorical(Y_test, 2)


# Load CNN and output features from layer fc15 for train set
model = load_model('ava_vgg_1.0_10.h5')
model.pop()
model.pop()
out = model.predict(X_train)
np.save("bottlenecked_features.npy",out)

# Load CNN and output features from layer fc15 for test set
out_test = model.predict(X_test) 
np.save("bottlenecked_features_test.npy",out_test)


out = np.load("bottlenecked_features.npy")
out_test = np.load("bottlenecked_features_test.npy")


classifier_filename = "classifier_CNNSVM.p"
classifier = pickle.load(open(classifier_filename, "rb"))
classifier = train(out, Y_train)
pickle.dump(classifier, open(classifier_filename, "wb"))
accuracy_score = accuracy_score(Y_test, classifier.predict(out_test))

confidence = classifier.decision_function(out_test)

predicted_values = classifier.predict(out_test)

ava_test.loc[:, 'confidence'] = confidence
ava_test.loc[:, 'predicted'] = predicted_values

store_full = HDFStore('dataset_h5/full_test_labels.h5')
store_full['full_labels_test_cnnsvm'] = ava_test


def train(features, labels):
    X = features
    Y = labels

    clf = svm.LinearSVC()
    #clf = linear_model.SGDRegressor()
    clf.fit(X, Y)

    return clf

def benchmark():
    extract_model = model
    extract_model.pop()
    extract_model.pop()
    %time model.predict(X_train[:100]) 