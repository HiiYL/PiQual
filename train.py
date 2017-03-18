from googlenet_custom_layers import PoolHelper,LRN
from keras.models import model_from_json

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

from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import  Flatten, merge, Reshape, Activation
from keras.layers import Input, LSTM, Dense, Activation, GRU,Convolution1D,Dropout
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D,GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop,Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.image import ImageDataGenerator


from googlenet_custom_layers import PoolHelper,LRN

from datetime import datetime

from utils.utils import process_image, deprocess_image
from utils.utils import read_and_generate_heatmap, prepare_data,evaluate_distribution_accuracy

from models import create_model

max_features = 20000
maxlen=100
EMBEDDING_DIM = 300

use_distribution = True
use_semantics = False
# X_train, Y_train, X_test, Y_test = prepare_data(use_distribution=use_distribution)
X_train, Y_train,X_test, Y_test,X_train_text, X_test_text,embedding_layer =
 prepare_data(use_distribution=use_distribution, use_semantics=use_semantics, use_comments=True)
# X_train, Y_train,X_test, Y_test= prepare_data(use_distribution=use_distribution, use_semantics=False)
# X_train, Y_train, Y_train_semantics, X_test, Y_test, Y_test_semantics, X_train_text, X_test_text, embedding_layer = prepare_data(use_distribution=use_distribution, use_semantics=use_semantics, use_comments=True)


## Without image data
# _, Y_train,_, Y_test,X_train_text, X_test_text,embedding_layer = prepare_data(use_distribution=use_distribution, use_semantics=use_semantics, use_comments=True, imageDataAvailable=False)



# BEST MODEL
model = create_model('weights/2017-01-25 22_56_09 - distribution_2layergru_extra_conv_layer.h5',
 use_distribution=use_distribution, use_semantics=use_semantics,use_multigap=True,use_comments=True,
  embedding_layer=embedding_layer,extra_conv_layer=True,textInputMaxLength=maxlen,embedding_dim=EMBEDDING_DIM)

# model = create_model('weights/googlenet_aesthetics_weights.h5',
#  use_distribution=use_distribution, use_semantics=use_semantics,use_multigap=True, heatmap=False)

# MODEL WITH EXTRA CONV AND NO TEXT
# model = create_model('weights/2017-01-27 12:41:36 - distribution_extra_conv_layer.h5',
#  use_distribution=use_distribution, use_semantics=use_semantics,
#  use_multigap=True,extra_conv_layer=True)


# RAPID STYLE
# model = create_model('weights/googlenet_aesthetics_weights.h5',
#  use_distribution=use_distribution, use_semantics=True,
#  use_multigap=False,extra_conv_layer=False, rapid_style=True)


# rmsProp = RMSprop(lr=0.0001,clipnorm=1.,clipvalue=0.5)
adam = Adam(lr=0.0001,clipnorm=1.,clipvalue=0.5)

if use_distribution:
    print("using kld loss...")
    model.compile(optimizer=adam,loss='kld', metrics=['accuracy'])#,loss_weights=[1., 0.2])
else:
    print("using categorical crossentropy loss...")
    model.compile(optimizer=adam,loss='categorical_crossentropy', metrics=['accuracy'])#,loss_weights=[1., 0.2])


time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

model_identifier = "joint_distribution_gru_singegap"
unique_model_identifier = "{} - {}".format(time_now, model_identifier)

checkpointer = ModelCheckpoint(filepath="weights/{}.h5".format(unique_model_identifier), verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=3)
csv_logger = CSVLogger('logs/{}.log'.format(unique_model_identifier))

# model.fit([X_train,X_train_text],Y_train,
#     nb_epoch=20, batch_size=32, shuffle="batch",
#     validation_data=([X_test,X_test_text], Y_test),
#     callbacks=[csv_logger,checkpointer,reduce_lr])#,reduce_lr])#,class_weight = class_weight)


# model.fit(X_train,Y_train,
#     nb_epoch=20, batch_size=32, shuffle="batch",
#     validation_data=(X_test, Y_test),
#     callbacks=[csv_logger,checkpointer,reduce_lr])#,reduce_lr])#,class_weight = class_weight)


model.fit([X_train,X_train_text],Y_train,
    nb_epoch=20, batch_size=32, shuffle="batch",
    validation_data=([X_test,X_test_text], Y_test),
    callbacks=[csv_logger,checkpointer,reduce_lr])#,reduce_lr])#,class_weight = class_weight)

accuracy = evaluate_distribution_accuracy(model, [X_test,X_test_text], Y_test)
print(accuracy)