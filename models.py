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

from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import  Flatten, merge, Reshape, Activation
from keras.layers import Input, Embedding, LSTM, Dense, Activation, GRU,Convolution1D,Dropout
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

def create_model(weights_path=None, use_distribution=False, use_multigap=False,
    use_semantics=False, rapid_style=False, use_comments=False, embedding_layer=None,
     extra_conv_layer=False,load_weights_by_name=True,textInputMaxLength=100, embedding_dim=300):

    input_image = Input(shape=(3, 224, 224))

    if rapid_style:
        if not use_semantics:
            print("[WARN] Semantics is not enabled, rapid style parameter will be ignored")

    if use_comments:
        if embedding_layer is None:
            print("[ERROR] Embedding layer is required for creating comments model")
            return None

        comment_input = Input(shape=(textInputMaxLength,), dtype='int32')
        embedded_sequences = embedding_layer(comment_input)
        
        x_text_aesthetics = GRU(embedding_dim,dropout_W = 0.3,dropout_U = 0.3,return_sequences=True, name='gru_aesthetics_1')(embedded_sequences)
        x_text_aesthetics = GRU(embedding_dim,dropout_W = 0.3,dropout_U = 0.3,name='gru_aesthetics_2')(x_text_aesthetics)

        if use_semantics and not rapid_style:
            x_text_semantics = GRU(embedding_dim,dropout_W = 0.3,dropout_U = 0.3,return_sequences=True, name='gru_semantics_1')(embedded_sequences)
            x_text_semantics = GRU(embedding_dim,dropout_W = 0.3,dropout_U = 0.3, name='gru_semantics_2')(x_text_semantics)

    
    conv1_7x7_s2 = Convolution2D(64,7,7,subsample=(2,2),border_mode='same',activation='relu',name='conv1/7x7_s2',W_regularizer=l2(0.0002))(input_image)
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)
    pool1_helper = PoolHelper()(conv1_zero_pad)
    pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool1/3x3_s2')(pool1_helper)
    pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2)
    conv2_3x3_reduce = Convolution2D(64,1,1,border_mode='same',activation='relu',name='conv2/3x3_reduce',W_regularizer=l2(0.0002))(pool1_norm1)
    conv2_3x3 = Convolution2D(192,3,3,border_mode='same',activation='relu',name='conv2/3x3',W_regularizer=l2(0.0002))(conv2_3x3_reduce)
    conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3)
    conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)
    pool2_helper = PoolHelper()(conv2_zero_pad)
    pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool2/3x3_s2')(pool2_helper)
    
    
    inception_3a_1x1 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_3a/1x1',W_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_3x3_reduce = Convolution2D(96,1,1,border_mode='same',activation='relu',name='inception_3a/3x3_reduce',W_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_3x3 = Convolution2D(128,3,3,border_mode='same',activation='relu',name='inception_3a/3x3',W_regularizer=l2(0.0002))(inception_3a_3x3_reduce)
    inception_3a_5x5_reduce = Convolution2D(16,1,1,border_mode='same',activation='relu',name='inception_3a/5x5_reduce',W_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_5x5 = Convolution2D(32,5,5,border_mode='same',activation='relu',name='inception_3a/5x5',W_regularizer=l2(0.0002))(inception_3a_5x5_reduce)
    inception_3a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_3a/pool')(pool2_3x3_s2)
    inception_3a_pool_proj = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_3a/pool_proj',W_regularizer=l2(0.0002))(inception_3a_pool)
    inception_3a_output = merge([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj],mode='concat',concat_axis=1,name='inception_3a/output')

    inception_3b_1x1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_3b/1x1',W_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_3x3_reduce = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_3b/3x3_reduce',W_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_3x3 = Convolution2D(192,3,3,border_mode='same',activation='relu',name='inception_3b/3x3',W_regularizer=l2(0.0002))(inception_3b_3x3_reduce)
    inception_3b_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_3b/5x5_reduce',W_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_5x5 = Convolution2D(96,5,5,border_mode='same',activation='relu',name='inception_3b/5x5',W_regularizer=l2(0.0002))(inception_3b_5x5_reduce)
    inception_3b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_3b/pool')(inception_3a_output)
    inception_3b_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_3b/pool_proj',W_regularizer=l2(0.0002))(inception_3b_pool)
    inception_3b_output = merge([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj],mode='concat',concat_axis=1,name='inception_3b/output')
    
    inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)
    pool3_helper = PoolHelper()(inception_3b_output_zero_pad)
    pool3_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool3/3x3_s2')(pool3_helper)
    
    
    inception_4a_1x1 = Convolution2D(192,1,1,border_mode='same',activation='relu',name='inception_4a/1x1',W_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_3x3_reduce = Convolution2D(96,1,1,border_mode='same',activation='relu',name='inception_4a/3x3_reduce',W_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_3x3 = Convolution2D(208,3,3,border_mode='same',activation='relu',name='inception_4a/3x3',W_regularizer=l2(0.0002))(inception_4a_3x3_reduce)
    inception_4a_5x5_reduce = Convolution2D(16,1,1,border_mode='same',activation='relu',name='inception_4a/5x5_reduce',W_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_5x5 = Convolution2D(48,5,5,border_mode='same',activation='relu',name='inception_4a/5x5',W_regularizer=l2(0.0002))(inception_4a_5x5_reduce)
    inception_4a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4a/pool')(pool3_3x3_s2)
    inception_4a_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4a/pool_proj',W_regularizer=l2(0.0002))(inception_4a_pool)
    inception_4a_output = merge([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj],mode='concat',concat_axis=1,name='inception_4a/output')

    if extra_conv_layer:
        conv_4a_output = Convolution2D(624, 3, 3, activation='relu',name='conv_4a',border_mode = 'same',W_regularizer=l2(0.0002))(inception_4a_output)
        inception_4a_gap = GlobalAveragePooling2D()(conv_4a_output)


    inception_4b_1x1 = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_4b/1x1',W_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_3x3_reduce = Convolution2D(112,1,1,border_mode='same',activation='relu',name='inception_4b/3x3_reduce',W_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_3x3 = Convolution2D(224,3,3,border_mode='same',activation='relu',name='inception_4b/3x3',W_regularizer=l2(0.0002))(inception_4b_3x3_reduce)
    inception_4b_5x5_reduce = Convolution2D(24,1,1,border_mode='same',activation='relu',name='inception_4b/5x5_reduce',W_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4b/5x5',W_regularizer=l2(0.0002))(inception_4b_5x5_reduce)
    inception_4b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4b/pool')(inception_4a_output)
    inception_4b_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4b/pool_proj',W_regularizer=l2(0.0002))(inception_4b_pool)
    inception_4b_output = merge([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj],mode='concat',concat_axis=1,name='inception_4b_output')

    if extra_conv_layer:
        conv_4b_output = Convolution2D(648, 3, 3, activation='relu',name='conv_4b',border_mode = 'same',W_regularizer=l2(0.0002))(inception_4b_output)
        inception_4b_gap = GlobalAveragePooling2D()(conv_4b_output)



    inception_4c_1x1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4c/1x1',W_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_3x3_reduce = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4c/3x3_reduce',W_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_3x3 = Convolution2D(256,3,3,border_mode='same',activation='relu',name='inception_4c/3x3',W_regularizer=l2(0.0002))(inception_4c_3x3_reduce)
    inception_4c_5x5_reduce = Convolution2D(24,1,1,border_mode='same',activation='relu',name='inception_4c/5x5_reduce',W_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4c/5x5',W_regularizer=l2(0.0002))(inception_4c_5x5_reduce)
    inception_4c_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4c/pool')(inception_4b_output)
    inception_4c_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4c/pool_proj',W_regularizer=l2(0.0002))(inception_4c_pool)
    inception_4c_output = merge([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj],mode='concat',concat_axis=1,name='inception_4c/output')

    if extra_conv_layer:
        conv_4c_output = Convolution2D(663, 3, 3, activation='relu',name='conv_4c',border_mode = 'same',W_regularizer=l2(0.0002))(inception_4c_output)
        inception_4c_gap = GlobalAveragePooling2D()(conv_4c_output)




    inception_4d_1x1 = Convolution2D(112,1,1,border_mode='same',activation='relu',name='inception_4d/1x1',W_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_3x3_reduce = Convolution2D(144,1,1,border_mode='same',activation='relu',name='inception_4d/3x3_reduce',W_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_3x3 = Convolution2D(288,3,3,border_mode='same',activation='relu',name='inception_4d/3x3',W_regularizer=l2(0.0002))(inception_4d_3x3_reduce)
    inception_4d_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_4d/5x5_reduce',W_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4d/5x5',W_regularizer=l2(0.0002))(inception_4d_5x5_reduce)
    inception_4d_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4d/pool')(inception_4c_output)
    inception_4d_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4d/pool_proj',W_regularizer=l2(0.0002))(inception_4d_pool)
    inception_4d_output = merge([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj],mode='concat',concat_axis=1,name='inception_4d/output')

    if extra_conv_layer:
        conv_4d_output = Convolution2D(704, 3, 3, activation='relu',name='conv_4d',border_mode = 'same',W_regularizer=l2(0.0002))(inception_4d_output)
        inception_4d_gap = GlobalAveragePooling2D()(conv_4d_output)

    if use_semantics:
        inception_4e_1x1_aesthetics = Convolution2D(256,1,1,border_mode='same',activation='relu',name='inception_4e/1x1_aesthetics',W_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_3x3_reduce_aesthetics = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_4e/3x3_reduce_aesthetics',W_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_3x3_aesthetics= Convolution2D(320,3,3,border_mode='same',activation='relu',name='inception_4e/3x3_aesthetics',W_regularizer=l2(0.0002))(inception_4e_3x3_reduce_aesthetics)
        inception_4e_5x5_reduce_aesthetics = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_4e/5x5_reduce_aesthetics',W_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_5x5_aesthetics = Convolution2D(128,5,5,border_mode='same',activation='relu',name='inception_4e/5x5_aesthetics',W_regularizer=l2(0.0002))(inception_4e_5x5_reduce_aesthetics)
        inception_4e_pool_aesthetics = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4e/pool_aesthetics')(inception_4d_output)
        inception_4e_pool_proj_aesthetics = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4e/pool_proj_aesthetics',W_regularizer=l2(0.0002))(inception_4e_pool_aesthetics)
        inception_4e_output_aesthetics = merge([inception_4e_1x1_aesthetics,inception_4e_3x3_aesthetics,inception_4e_5x5_aesthetics,inception_4e_pool_proj_aesthetics],mode='concat',concat_axis=1,name='inception_4e/output_aesthetics')
        conv_output_aesthetics = Convolution2D(1024, 3, 3, activation='relu',name='conv_6_1',border_mode = 'same',W_regularizer=l2(0.0002))(inception_4e_output_aesthetics)
        
        if rapid_style:
            conv1_7x7_s2 = Convolution2D(64,7,7,subsample=(2,2),border_mode='same',activation='relu',name='semantic_conv1/7x7_s2',W_regularizer=l2(0.0002))(input_image)
            conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)
            pool1_helper = PoolHelper()(conv1_zero_pad)
            pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='semantic_pool1/3x3_s2')(pool1_helper)
            pool1_norm1 = LRN(name='semantic_pool1/norm1')(pool1_3x3_s2)
            conv2_3x3_reduce = Convolution2D(64,1,1,border_mode='same',activation='relu',name='semantic_conv2/3x3_reduce',W_regularizer=l2(0.0002))(pool1_norm1)
            conv2_3x3 = Convolution2D(192,3,3,border_mode='same',activation='relu',name='semantic_conv2/3x3',W_regularizer=l2(0.0002))(conv2_3x3_reduce)
            conv2_norm2 = LRN(name='semantic_conv2/norm2')(conv2_3x3)
            conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)
            pool2_helper = PoolHelper()(conv2_zero_pad)
            pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='semantic_pool2/3x3_s2')(pool2_helper)
            
            inception_3a_1x1 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='semantic_inception_3a/1x1',W_regularizer=l2(0.0002))(pool2_3x3_s2)
            inception_3a_3x3_reduce = Convolution2D(96,1,1,border_mode='same',activation='relu',name='semantic_inception_3a/3x3_reduce',W_regularizer=l2(0.0002))(pool2_3x3_s2)
            inception_3a_3x3 = Convolution2D(128,3,3,border_mode='same',activation='relu',name='semantic_inception_3a/3x3',W_regularizer=l2(0.0002))(inception_3a_3x3_reduce)
            inception_3a_5x5_reduce = Convolution2D(16,1,1,border_mode='same',activation='relu',name='semantic_inception_3a/5x5_reduce',W_regularizer=l2(0.0002))(pool2_3x3_s2)
            inception_3a_5x5 = Convolution2D(32,5,5,border_mode='same',activation='relu',name='semantic_inception_3a/5x5',W_regularizer=l2(0.0002))(inception_3a_5x5_reduce)
            inception_3a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='semantic_inception_3a/pool')(pool2_3x3_s2)
            inception_3a_pool_proj = Convolution2D(32,1,1,border_mode='same',activation='relu',name='semantic_inception_3a/pool_proj',W_regularizer=l2(0.0002))(inception_3a_pool)
            inception_3a_output = merge([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj],mode='concat',concat_axis=1,name='semantic_inception_3a/output')
              
            inception_3b_1x1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='semantic_inception_3b/1x1',W_regularizer=l2(0.0002))(inception_3a_output)
            inception_3b_3x3_reduce = Convolution2D(128,1,1,border_mode='same',activation='relu',name='semantic_inception_3b/3x3_reduce',W_regularizer=l2(0.0002))(inception_3a_output)
            inception_3b_3x3 = Convolution2D(192,3,3,border_mode='same',activation='relu',name='semantic_inception_3b/3x3',W_regularizer=l2(0.0002))(inception_3b_3x3_reduce)
            inception_3b_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='semantic_inception_3b/5x5_reduce',W_regularizer=l2(0.0002))(inception_3a_output)
            inception_3b_5x5 = Convolution2D(96,5,5,border_mode='same',activation='relu',name='semantic_inception_3b/5x5',W_regularizer=l2(0.0002))(inception_3b_5x5_reduce)
            inception_3b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='semantic_inception_3b/pool')(inception_3a_output)
            inception_3b_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='semantic_inception_3b/pool_proj',W_regularizer=l2(0.0002))(inception_3b_pool)
            inception_3b_output = merge([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj],mode='concat',concat_axis=1,name='semantic_inception_3b/output')
            
            inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)
            pool3_helper = PoolHelper()(inception_3b_output_zero_pad)
            pool3_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='semantic_pool3/3x3_s2')(pool3_helper)
            
            inception_4a_1x1 = Convolution2D(192,1,1,border_mode='same',activation='relu',name='semantic_inception_4a/1x1',W_regularizer=l2(0.0002))(pool3_3x3_s2)
            inception_4a_3x3_reduce = Convolution2D(96,1,1,border_mode='same',activation='relu',name='semantic_inception_4a/3x3_reduce',W_regularizer=l2(0.0002))(pool3_3x3_s2)
            inception_4a_3x3 = Convolution2D(208,3,3,border_mode='same',activation='relu',name='semantic_inception_4a/3x3',W_regularizer=l2(0.0002))(inception_4a_3x3_reduce)
            inception_4a_5x5_reduce = Convolution2D(16,1,1,border_mode='same',activation='relu',name='semantic_inception_4a/5x5_reduce',W_regularizer=l2(0.0002))(pool3_3x3_s2)
            inception_4a_5x5 = Convolution2D(48,5,5,border_mode='same',activation='relu',name='semantic_inception_4a/5x5',W_regularizer=l2(0.0002))(inception_4a_5x5_reduce)
            inception_4a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='semantic_inception_4a/pool')(pool3_3x3_s2)
            inception_4a_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='semantic_inception_4a/pool_proj',W_regularizer=l2(0.0002))(inception_4a_pool)
            inception_4a_output = merge([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj],mode='concat',concat_axis=1,name='semantic_inception_4a/output')
             
            inception_4b_1x1 = Convolution2D(160,1,1,border_mode='same',activation='relu',name='semantic_inception_4b/1x1',W_regularizer=l2(0.0002))(inception_4a_output)
            inception_4b_3x3_reduce = Convolution2D(112,1,1,border_mode='same',activation='relu',name='semantic_inception_4b/3x3_reduce',W_regularizer=l2(0.0002))(inception_4a_output)
            inception_4b_3x3 = Convolution2D(224,3,3,border_mode='same',activation='relu',name='semantic_inception_4b/3x3',W_regularizer=l2(0.0002))(inception_4b_3x3_reduce)
            inception_4b_5x5_reduce = Convolution2D(24,1,1,border_mode='same',activation='relu',name='semantic_inception_4b/5x5_reduce',W_regularizer=l2(0.0002))(inception_4a_output)
            inception_4b_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='semantic_inception_4b/5x5',W_regularizer=l2(0.0002))(inception_4b_5x5_reduce)
            inception_4b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='semantic_inception_4b/pool')(inception_4a_output)
            inception_4b_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='semantic_inception_4b/pool_proj',W_regularizer=l2(0.0002))(inception_4b_pool)
            inception_4b_output = merge([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj],mode='concat',concat_axis=1,name='semantic_inception_4b_output')
            
            
            inception_4c_1x1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='semantic_inception_4c/1x1',W_regularizer=l2(0.0002))(inception_4b_output)
            inception_4c_3x3_reduce = Convolution2D(128,1,1,border_mode='same',activation='relu',name='semantic_inception_4c/3x3_reduce',W_regularizer=l2(0.0002))(inception_4b_output)
            inception_4c_3x3 = Convolution2D(256,3,3,border_mode='same',activation='relu',name='semantic_inception_4c/3x3',W_regularizer=l2(0.0002))(inception_4c_3x3_reduce)
            inception_4c_5x5_reduce = Convolution2D(24,1,1,border_mode='same',activation='relu',name='semantic_inception_4c/5x5_reduce',W_regularizer=l2(0.0002))(inception_4b_output)
            inception_4c_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='semantic_inception_4c/5x5',W_regularizer=l2(0.0002))(inception_4c_5x5_reduce)
            inception_4c_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='semantic_inception_4c/pool')(inception_4b_output)
            inception_4c_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='semantic_inception_4c/pool_proj',W_regularizer=l2(0.0002))(inception_4c_pool)
            inception_4c_output = merge([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj],mode='concat',concat_axis=1,name='semantic_inception_4c/output')
            
            
            inception_4d_1x1 = Convolution2D(112,1,1,border_mode='same',activation='relu',name='semantic_inception_4d/1x1',W_regularizer=l2(0.0002))(inception_4c_output)
            inception_4d_3x3_reduce = Convolution2D(144,1,1,border_mode='same',activation='relu',name='semantic_inception_4d/3x3_reduce',W_regularizer=l2(0.0002))(inception_4c_output)
            inception_4d_3x3 = Convolution2D(288,3,3,border_mode='same',activation='relu',name='semantic_inception_4d/3x3',W_regularizer=l2(0.0002))(inception_4d_3x3_reduce)
            inception_4d_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='semantic_inception_4d/5x5_reduce',W_regularizer=l2(0.0002))(inception_4c_output)
            inception_4d_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='semantic_inception_4d/5x5',W_regularizer=l2(0.0002))(inception_4d_5x5_reduce)
            inception_4d_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='semantic_inception_4d/pool')(inception_4c_output)
            inception_4d_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='semantic_inception_4d/pool_proj',W_regularizer=l2(0.0002))(inception_4d_pool)
            inception_4d_output = merge([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj],mode='concat',concat_axis=1,name='semantic_inception_4d/output')
            

        inception_4e_1x1_semantics = Convolution2D(256,1,1,border_mode='same',activation='relu',name='inception_4e/1x1_semantics',W_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_3x3_reduce_semantics = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_4e/3x3_reduce_semantics',W_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_3x3_semantics = Convolution2D(320,3,3,border_mode='same',activation='relu',name='inception_4e/3x3_semantics',W_regularizer=l2(0.0002))(inception_4e_3x3_reduce_semantics)
        inception_4e_5x5_reduce_semantics = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_4e/5x5_reduce_semantics',W_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_5x5_semantics = Convolution2D(128,5,5,border_mode='same',activation='relu',name='inception_4e/5x5_semantics',W_regularizer=l2(0.0002))(inception_4e_5x5_reduce_semantics)
        inception_4e_pool_semantics = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4e/pool_semantics')(inception_4d_output)
        inception_4e_pool_proj_semantics = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4e/pool_proj_semantics',W_regularizer=l2(0.0002))(inception_4e_pool_semantics)
        inception_4e_output_semantics = merge([inception_4e_1x1_semantics,inception_4e_3x3_semantics,inception_4e_5x5_semantics,inception_4e_pool_proj_semantics],mode='concat',concat_axis=1,name='inception_4e/output_semantics')
        conv_output_semantics = Convolution2D(1024, 3, 3, activation='relu',name='conv_6_1_semantics',border_mode = 'same',W_regularizer=l2(0.0002))(inception_4e_output_semantics)
        

        if not rapid_style:
            x_semantics = GlobalAveragePooling2D()(conv_output_semantics)
            if use_comments:
                 x_semantics = merge([x_semantics, x_text_semantics],mode='concat',concat_axis=1)
            output_semantics = Dense(65, activation = 'softmax', name="output_semantics")(x_semantics)

    else:
        inception_4e_1x1 = Convolution2D(256,1,1,border_mode='same',activation='relu',name='inception_4e/1x1',W_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_3x3_reduce = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_4e/3x3_reduce',W_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_3x3 = Convolution2D(320,3,3,border_mode='same',activation='relu',name='inception_4e/3x3',W_regularizer=l2(0.0002))(inception_4e_3x3_reduce)
        inception_4e_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_4e/5x5_reduce',W_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_5x5 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='inception_4e/5x5',W_regularizer=l2(0.0002))(inception_4e_5x5_reduce)
        inception_4e_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4e/pool')(inception_4d_output)
        inception_4e_pool_proj = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4e/pool_proj',W_regularizer=l2(0.0002))(inception_4e_pool)
        inception_4e_output = merge([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj],mode='concat',concat_axis=1,name='inception_4e/output')  
        conv_output_aesthetics = Convolution2D(1024, 3, 3, activation='relu',name='conv_6_1',border_mode = 'same',W_regularizer=l2(0.0002))(inception_4e_output)
    
    if rapid_style:
        merged_conv = merge([conv_output_aesthetics, conv_output_semantics], mode='concat', concat_axis=1)
        x_aesthetics =  GlobalAveragePooling2D()(merged_conv)
    else:
        x_aesthetics = GlobalAveragePooling2D()(conv_output_aesthetics)

    if use_multigap:
        if use_comments:
            x_aesthetics = merge([x_aesthetics, x_text_aesthetics, inception_4a_gap, inception_4b_gap, inception_4c_gap, inception_4d_gap],mode='concat',concat_axis=1)
        else:
            x_aesthetics = merge([x_aesthetics, inception_4a_gap, inception_4b_gap, inception_4c_gap, inception_4d_gap],mode='concat',concat_axis=1)
    else:
        if use_comments:
            x_aesthetics = merge([x_aesthetics, x_text_aesthetics],mode='concat',concat_axis=1)


    if use_distribution:
        if use_multigap:
            output_aesthetics = Dense(10, activation = 'softmax', name="main_output__")(x_aesthetics)
        else:
            output_aesthetics = Dense(10, activation = 'softmax', name="main_output_")(x_aesthetics)
    else:
        if use_multigap:
            output_aesthetics = Dense(2, activation = 'softmax', name="main_output_")(x_aesthetics)
        else:
            output_aesthetics = Dense(2, activation = 'softmax', name="main_output")(x_aesthetics)
    
    if use_semantics and not rapid_style:
        if use_comments:
            googlenet = Model(input=[input_image, comment_input], output=[output_aesthetics,output_semantics])
        else:
            googlenet = Model(input=input_image, output=[output_aesthetics,output_semantics])
    else:
        if use_comments:
            googlenet = Model(input=[input_image, comment_input], output=output_aesthetics)
        else:
            googlenet = Model(input=input_image, output=output_aesthetics)
    
    if weights_path:
        if use_semantics:
            googlenet.load_weights('weights/named_googlenet_semantics_weights.h5', by_name=True)
        googlenet.load_weights(weights_path,by_name=load_weights_by_name)

        if rapid_style:
            for i, layer in enumerate(googlenet.layers):
                if 'semantic' in layer.name:
                    # print("{} - {}".format(i, layer.name))
                    layer.trainable = False

    return googlenet