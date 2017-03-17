import numpy as np
import h5py
import os
import cv2
from sklearn.decomposition import PCA
import ggmm.gpu as ggmm

from sklearn import svm
from sklearn.metrics import accuracy_score


import pandas as pd
from pandas import HDFStore, DataFrame
import pickle

from image_fisher_vector import ImageFisherVector as ifv

classifier =  pickle.load( open( "classifier.p", "rb" ) )
print("Loaded Classifier!")
gmm = ifv.load_gmm()


image = cv2.imread("model.png")
image_features = ifv.extract_image_features(image)

if image_features is not None and image_features.shape[0] >= 64:
	image_features = ifv.reduce_features(image_features)
else:
	print("Not enough features found by SIFT")


fv = fisher_vector(image_features,gmm)

classifier.decision_function(fv.reshape(1, -1))

confidence = classifier.decision_function(fv_test)

labels_test = load_full_labels(skipped_indices,labels_to_test, False)
labels_test['confidence'] = confidence

top_confidence_good = labels_test.ix[labels_test.good ==1 ].sort('confidence', ascending=False)[:10]

for index, row in top_confidence_good.iterrows():
    filename = "{0}.jpg".format(index)

    filepath = os.path.join(ava_data_path, filename)
    image = cv2.imread(filepath)
    top
