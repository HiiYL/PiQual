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
use_comments=True
use_multigap=True
# X_train, Y_train, X_test, Y_test = prepare_data(use_distribution=use_distribution)
X_train, Y_train,X_test, Y_test,X_train_text, X_test_text,embedding_layer =
 prepare_data(use_distribution=use_distribution, use_semantics=use_semantics, use_comments=use_comments)
# X_train, Y_train,X_test, Y_test= prepare_data(use_distribution=use_distribution, use_semantics=False)
# X_train, Y_train, Y_train_semantics, X_test, Y_test, Y_test_semantics, X_train_text, X_test_text, embedding_layer = prepare_data(use_distribution=use_distribution, use_semantics=use_semantics, use_comments=use_comments)


## Without image data
# _, Y_train,_, Y_test,X_train_text, X_test_text,embedding_layer = prepare_data(use_distribution=use_distribution, use_semantics=use_semantics, use_comments=use_comments, imageDataAvailable=False)



# BEST MODEL
model = create_model('weights/2017-01-25 22_56_09 - distribution_2layergru_extra_conv_layer.h5',
 use_distribution=use_distribution, use_semantics=use_semantics,use_multigap=use_multigap,use_comments=use_comments,
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

from keras.utils.visualize_util import plot
plot(model, to_file='{}.png'.format(unique_model_identifier),show_shapes=True)

from keras import backend as K
def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

gap_conv_layer_4a = get_output_layer(model, "conv_4a")
gap_conv_layer_4b = get_output_layer(model, "conv_4b")
gap_conv_layer_4c = get_output_layer(model, "conv_4c")
gap_conv_layer_4d = get_output_layer(model, "conv_4d")

final_conv_layer = get_output_layer(model, "conv_6_1")

if use_multigap:
    if use_comments:
        get_output = K.function( 
            [ model.inputs[0], model.inputs[1],K.learning_phase() ] ,
             [final_conv_layer.output,gap_conv_layer_4a.output,
             gap_conv_layer_4b.output,gap_conv_layer_4c.output,
             gap_conv_layer_4d.output, model.layers[-1].output])
    else:
        get_output = K.function( 
            [ model.inputs[0],K.learning_phase() ] ,
             [final_conv_layer.output,gap_conv_layer_4a.output,
             gap_conv_layer_4b.output,gap_conv_layer_4c.output,
             gap_conv_layer_4d.output, model.layers[-1].output])
else:
    get_output = K.function( 
        [ model.inputs[0],K.learning_phase() ] ,
         [final_conv_layer.output, model.layers[-1].output])


class_weights = model.layers[-1].get_weights()[0]

images_to_show = 50


### To show a range of values
#
# total_amount = X_test_text.shape[0]
# middle = int(total_amount/2)
# min_boundary = middle - int(images_to_show/2)
# max_boundary = middle + int(images_to_show/2)
# X_test_text_used = X_test_text[min_boundary:max_boundary]#[::-1]
#
###


class_weights_to_visualize = class_weights[1324:1948]
class_weights_to_visualize =  np.column_stack((
    class_weights_to_visualize[:,0:5].mean(axis=1),
     class_weights_to_visualize[:,5:10].mean(axis=1)))


X_test_text_used = X_test_text[-images_to_show:][::-1]
for comment_idx, index in enumerate(ava_test[-images_to_show:][::-1].index):

    output_filename = "heatmaps/{} - comments.png".format(index)

    if os.path.isfile(output_filename):
        print("[INFO] file with id of {} already exists, skipping".format(index))
    else:
        input_path = "datasetdataset/AVA/data/{}.jpg".format(index)
        original_img = cv2.imread(input_path).astype(np.float32)

        width, height,_ = original_img.shape
        original_img = cv2.resize(original_img, (int(height / 2),int(width /2)))
        width, height,_ = original_img.shape

        im = process_image(cv2.resize(original_img,(224,224)))

        [conv_outputs, gap_conv_outputs_4a,gap_conv_outputs_4b,
        gap_conv_outputs_4c,gap_conv_outputs_4d, predictions] = get_output( [im,
            np.expand_dims(X_test_text_used[comment_idx], axis=0),
             0])

        # [conv_outputs, predictions] = get_output( [im,0])


        conv_to_visualize = gap_conv_outputs_4a[0, :, :, :]
        # conv_to_visualize = conv_outputs[0, :, :, :]

        output_image = original_img.copy()

        for class_weight_to_visualize in class_weights_to_visualize.T:
            cam = np.zeros(dtype = np.float32, shape = conv_to_visualize.shape[1:3])

            class_to_visualize = 1 # 0 for bad, 1 for good
            for i, w in enumerate(class_weight_to_visualize):
                    cam += w * conv_to_visualize[i, :, :]

            cam /= np.max(cam)
            cam = cv2.resize(cam, (height, width))
            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            heatmap[np.where(cam < 0.2)] = 0
            img_cam = heatmap*0.5 + original_img
            print("CALLED CONCATENATE")
            output_image = np.concatenate((output_image, img_cam), axis=1)

        # output_image = cv2.resize(output_image, (int(output_image.shape[1] / 2), int(output_image.shape[0]/2)))

        cv2.imwrite(output_filename, output_image)
        print()
        print()