from googlenet_custom_layers import PoolHelper,LRN
from keras.models import model_from_json

from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D,Convolution1D

from keras.layers import Dense, Activation
from keras.models import Model

from keras.utils.np_utils import to_categorical
from keras.callbacks import CSVLogger, ReduceLROnPlateau

from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D,GlobalMaxPooling1D

from keras.optimizers import SGD
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
from keras.layers import Input, Embedding, LSTM, Dense, Activation, GRU,Convolution1D,Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from googlenet_custom_layers import PoolHelper,LRN

from datetime import datetime

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_features = 20000
maxlen=100
batch_size = 64

hidden_dims = 250
nb_epoch = 100

EMBEDDING_DIM = 300
delta = 1.0

# Convolution
filter_length = 5
nb_filter = 64
pool_length = 4


GLOVE_DIR = "../comments/glove/"

def tokenizeAndGenerateIndex(train, test):
  merged = np.concatenate([train, test])
  tokenizer = Tokenizer(nb_words=max_features)
  tokenizer.fit_on_texts(merged)
  sequences_train = tokenizer.texts_to_sequences(train)
  sequences_test = tokenizer.texts_to_sequences(test)
  word_index = tokenizer.word_index
  print('Found %s unique tokens.' % len(word_index))
  data_train = pad_sequences(sequences_train, maxlen=maxlen)
  data_test = pad_sequences(sequences_test, maxlen=maxlen)
  return data_train, data_test, word_index


def generateIndexMappingToEmbedding():
  embeddings_index = {}
  f = open(os.path.join(GLOVE_DIR, 'glove.6B.{}d.txt'.format(EMBEDDING_DIM)))
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
  f.close()

  print('Found %s word vectors.' % len(embeddings_index))
  return embeddings_index

def process_image(image):
    im = np.copy(image)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im

def deprocess_image(image):
    im = np.copy(image)
    im[:,:,0] += 103.939
    im[:,:,1] += 116.779
    im[:,:,2] += 123.

    im = im.transpose((1,2,0))

    return im

def read_and_generate_heatmap(input_path, output_path):
    original_img = cv2.imread(input_path).astype(np.float32)

    width, height, _ = original_img.shape

    im = process_image(cv2.resize(original_img,(224,224)))
    out = model.predict(im)

    class_weights = model.layers[-1].get_weights()[0]
    print("predictions", out[0])

    conv_output = out[2][0,:,:,:]
    #Create the class activation map.
    cam = np.zeros(dtype = np.float32, shape = conv_output.shape[1:3])

    class_to_visualize = 1 # 0 for bad, 1 for good
    for i, w in enumerate(class_weights[:, class_to_visualize]):
            cam += w * conv_output[i, :, :]

    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    temp = heatmap*0.5 + original_img
    file_name = input_path.split("/")[-1].split(".jpg")[0]

    cv2.imwrite('output/' + input_path.split("/")[-1],original_img)
    cv2.imwrite('output/{}_aesthetics.jpg'.format(file_name), temp)


    top_3_hits = np.argsort(out[1], axis=1)[0][::-1][:3]
    print("predictions", top_3_hits)
    semantic_tags = [ semantics.ix[hit + 1].semantic[1:] for hit in top_3_hits] ##NOTE: Quick fix to remove space 


    conv_output = out[3][0,:,:,:]
    #Create the class activation map.
    nth_top_semantic = 0

    for nth_top_semantic in range(len(top_3_hits)):
        cam = np.zeros(dtype = np.float32, shape = conv_output.shape[1:3])
        class_to_visualize = top_3_hits[nth_top_semantic] # 0 for bad, 1 for good
        for i, w in enumerate(class_weights[:, class_to_visualize]):
                cam += w * conv_output[i, :, :]
                
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0
        temp = heatmap*0.5 + original_img
        file_name = input_path.split("/")[-1].split(".jpg")[0]
        cv2.imwrite('output/{}_{}.jpg'.format(file_name, semantic_tags[nth_top_semantic]), temp)


def generate_images_heatmap(model, ava_table, style_category=-1):
    input_images_dir = "images/"
    output_dir = "output/"

    semantics = pd.read_table('../dataset/AVA/tags.txt',delimiter="(\d+)",
     usecols=[1,2], index_col=0, header=None,names=['index','semantic'])

    if style_category != -1:
        style = pd.read_table('../dataset/AVA/style_image_lists/train.jpgl', index_col=0)
        tag = pd.read_table('../dataset/AVA/style_image_lists/train.lab')

        style.loc[:,'style'] = tag.as_matrix()
        ava_with_style = style.join(ava_table, how='inner')

        images_with_style = ava_with_style.ix[(ava_with_style.ix[:,'style'] == style_category)]
        images_with_style = images_with_style.sort_values(by="score")

        ava_path = "../dataset/AVA/data/"
        for index in images_with_style.iloc[::-1][:25].index:
            image_name = str(index) + ".jpg"
            input_path = ava_path + image_name
            output_path = output_dir + image_name
            read_and_generate_heatmap(input_path, output_path)
    else:
        for file in os.listdir(input_images_dir):
            if file.endswith('.jpg'):
                read_and_generate_heatmap(input_images_dir + file, output_dir + file)


def getDistribution(dataframe):
    ratings_matrix = dataframe.ix[:,:10]
    sum_of_ratings = (dataframe.ix[:,:10]).sum(axis=1)
    normalized_score_distribution = ratings_matrix.div(sum_of_ratings,axis='index')
    return normalized_score_distribution.as_matrix()


def prepareData(delta=0.0, use_distribution=False, use_semantics=False, use_comments=False):
    store = HDFStore('../dataset_h5/labels.h5','r')
    ava_table = store['labels_train']
    ava_test = store['labels_test']

    if delta > 0.0:
        ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]


    h5f = h5py.File('../dataset_h5/images_224_delta_{0}.h5'.format(delta),'r')
    X_train = h5f['data_train']
    X_test = h5f['data_test']

    if use_comments:
        comments_train = ava_table.ix[:,'comments'].as_matrix()
        comments_test = ava_test.ix[:,'comments'].as_matrix()

        X_train_text, X_test_text, word_index = tokenizeAndGenerateIndex(comments_train, comments_test)

        embeddings_index = generateIndexMappingToEmbedding()
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=maxlen,
                                    trainable=False)

    if use_distribution:
        Y_train = getDistribution(ava_table)
        Y_test = getDistribution(ava_test)
    else:
        Y_train = ava_table.ix[:, "good"].as_matrix()
        Y_train = to_categorical(Y_train, 2)

        Y_test = ava_test.ix[:, "good"].as_matrix()
        Y_test = to_categorical(Y_test, 2)

    if use_semantics:
        Y_train_semantics = to_categorical(ava_table.ix[:,10:12].as_matrix())[:,1:]
        Y_test_semantics = to_categorical(ava_test.ix[:,10:12].as_matrix())[:,1:]

        return X_train, Y_train, Y_train_semantics, X_test, Y_test, Y_test_semantics

    return X_train, Y_train, X_test, Y_test


def evaluate_distribution_accuracy(model, X_test, Y_test):
    aesthetics_pred = model.predict(X_test)

    weights = np.array([1,2,3,4,5,6,7,8,9,10])

    score_pred = (aesthetics_pred * weights).sum(axis=1)
    score_test = (Y_test * weights).sum(axis=1)

    Y_pred_binary = np.array([ 1 if row >= 5 else 0 for row in score_pred])
    Y_test_binary = np.array([ 1 if row >= 5 else 0 for row in score_test])


    accuracy = np.sum(Y_pred_binary == Y_test_binary) / len(Y_test_binary)

    print("accuracy = {} %".format(accuracy * 100))
    return accuracy



def create_googlenet(weights_path=None, use_distribution=False, use_multigap=False, heatmap=False):
    # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    input = Input(shape=(3, 224, 224))
    
    conv1_7x7_s2 = Convolution2D(64,7,7,subsample=(2,2),border_mode='same',activation='relu',name='conv1/7x7_s2',W_regularizer=l2(0.0002))(input)
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
    inception_4a_gap = GlobalAveragePooling2D()(inception_4a_output)


    inception_4b_1x1 = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_4b/1x1',W_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_3x3_reduce = Convolution2D(112,1,1,border_mode='same',activation='relu',name='inception_4b/3x3_reduce',W_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_3x3 = Convolution2D(224,3,3,border_mode='same',activation='relu',name='inception_4b/3x3',W_regularizer=l2(0.0002))(inception_4b_3x3_reduce)
    inception_4b_5x5_reduce = Convolution2D(24,1,1,border_mode='same',activation='relu',name='inception_4b/5x5_reduce',W_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4b/5x5',W_regularizer=l2(0.0002))(inception_4b_5x5_reduce)
    inception_4b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4b/pool')(inception_4a_output)
    inception_4b_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4b/pool_proj',W_regularizer=l2(0.0002))(inception_4b_pool)
    inception_4b_output = merge([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj],mode='concat',concat_axis=1,name='inception_4b_output')
    inception_4b_gap = GlobalAveragePooling2D()(inception_4b_output)


    inception_4c_1x1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4c/1x1',W_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_3x3_reduce = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4c/3x3_reduce',W_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_3x3 = Convolution2D(256,3,3,border_mode='same',activation='relu',name='inception_4c/3x3',W_regularizer=l2(0.0002))(inception_4c_3x3_reduce)
    inception_4c_5x5_reduce = Convolution2D(24,1,1,border_mode='same',activation='relu',name='inception_4c/5x5_reduce',W_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4c/5x5',W_regularizer=l2(0.0002))(inception_4c_5x5_reduce)
    inception_4c_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4c/pool')(inception_4b_output)
    inception_4c_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4c/pool_proj',W_regularizer=l2(0.0002))(inception_4c_pool)
    inception_4c_output = merge([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj],mode='concat',concat_axis=1,name='inception_4c/output')
    inception_4c_gap = GlobalAveragePooling2D()(inception_4c_output)


    inception_4d_1x1 = Convolution2D(112,1,1,border_mode='same',activation='relu',name='inception_4d/1x1',W_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_3x3_reduce = Convolution2D(144,1,1,border_mode='same',activation='relu',name='inception_4d/3x3_reduce',W_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_3x3 = Convolution2D(288,3,3,border_mode='same',activation='relu',name='inception_4d/3x3',W_regularizer=l2(0.0002))(inception_4d_3x3_reduce)
    inception_4d_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_4d/5x5_reduce',W_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4d/5x5',W_regularizer=l2(0.0002))(inception_4d_5x5_reduce)
    inception_4d_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4d/pool')(inception_4c_output)
    inception_4d_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4d/pool_proj',W_regularizer=l2(0.0002))(inception_4d_pool)
    inception_4d_output = merge([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj],mode='concat',concat_axis=1,name='inception_4d/output')
    inception_4d_gap = GlobalAveragePooling2D()(inception_4d_output)

    inception_4e_1x1 = Convolution2D(256,1,1,border_mode='same',activation='relu',name='inception_4e/1x1',W_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_3x3_reduce = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_4e/3x3_reduce',W_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_3x3 = Convolution2D(320,3,3,border_mode='same',activation='relu',name='inception_4e/3x3',W_regularizer=l2(0.0002))(inception_4e_3x3_reduce)
    inception_4e_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_4e/5x5_reduce',W_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_5x5 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='inception_4e/5x5',W_regularizer=l2(0.0002))(inception_4e_5x5_reduce)
    inception_4e_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4e/pool')(inception_4d_output)
    inception_4e_pool_proj = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4e/pool_proj',W_regularizer=l2(0.0002))(inception_4e_pool)
    inception_4e_output = merge([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj],mode='concat',concat_axis=1,name='inception_4e/output')
    
    conv_output_aesthetics = Convolution2D(1024, 3, 3, activation='relu',name='conv_6_1',border_mode = 'same',W_regularizer=l2(0.0002))(inception_4e_output)

    x_aesthetics = GlobalAveragePooling2D()(conv_output_aesthetics)

    if use_multigap:
        x_aesthetics = merge([x_aesthetics, inception_4a_gap, inception_4b_gap, inception_4c_gap, inception_4d_gap] ,mode='concat',concat_axis=1)

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
    
    googlenet = Model(input=input, output=output_aesthetics)
    
    if weights_path:
        googlenet.load_weights(weights_path,by_name=True)
    
    return googlenet


use_distribution = True
X_train, Y_train, X_test, Y_test = prepareData(use_distribution=use_distribution)

model = create_googlenet('weights/gap_distribution_weights_aes_only_sgd2016-12-31 14:53:03.h5', use_distribution=use_distribution,use_multigap=True,heatmap=False)
sgd = SGD(lr=0.001, decay=5e-4, momentum=0.9, nesterov=True)
# rmsProp = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)

if use_distribution:
    print("using kld loss...")
    model.compile(optimizer=sgd,loss='kld', metrics=['accuracy'])
else:
    print("using categorical crossentropy loss...")
    model.compile(optimizer=sgd,loss='categorical_crossentropy', metrics=['accuracy'])

time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
checkpointer = ModelCheckpoint(filepath="weights/gap_distribution_weights_aes_only_sgd{}.h5".format(time_now), verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=3)
csv_logger = CSVLogger('logs/gap_distribution_weights_aes_only_sgd{}.log'.format(time_now))

# class_weight = {0 : 0.67, 1: 0.33}
model.fit(X_train,Y_train,nb_epoch=20, batch_size=32, shuffle="batch",
 validation_data=(X_test, Y_test), callbacks=[csv_logger,checkpointer,reduce_lr])#,reduce_lr])#,class_weight = class_weight)

from keras.utils.visualize_util import plot
plot(model, to_file='model_multigap_distribution.png',show_shapes=True)



model = create_googlenet('weights/gap_distribution_weights_aes_only_sgd2016-12-31 14:53:03.h5',
 use_distribution=use_distribution,use_multigap=True,heatmap=False)

model.evaluate(X_test,Y_test)

accuracy = evaluate_distribution_accuracy(model, X_test, Y_test)









