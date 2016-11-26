from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Input, Dense, Activation
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.models import Model
from keras import backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import CSVLogger

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

def VGG_19_GAP_functional(weights_path=None,heatmap=False):

    inputs = Input(shape=(3, None, None))

    x = Convolution2D(64, 3, 3, activation='relu',name='conv1_1',border_mode ='same')(inputs)
    x = Convolution2D(64, 3, 3, activation='relu',name='conv1_2',border_mode = 'same')(x)
    x = MaxPooling2D((2,2), strides=(2,2), dim_ordering="th")(x)

    x = Convolution2D(128, 3, 3, activation='relu',name='conv2_1',border_mode = 'same')(x)
    x = Convolution2D(128, 3, 3, activation='relu',name='conv2_2',border_mode = 'same')(x)
    x = MaxPooling2D((2,2), strides=(2,2), dim_ordering="th")(x)

    x = Convolution2D(256, 3, 3, activation='relu',name='conv3_1',border_mode = 'same')(x)
    x = Convolution2D(256, 3, 3, activation='relu',name='conv3_2',border_mode = 'same')(x)
    x = Convolution2D(256, 3, 3, activation='relu',name='conv3_3',border_mode = 'same')(x)
    x = Convolution2D(256, 3, 3, activation='relu',name='conv3_4',border_mode = 'same')(x)
    x = MaxPooling2D((2,2), strides=(2,2), dim_ordering="th")(x)

    x = Convolution2D(512, 3, 3, activation='relu',name='conv4_1',border_mode = 'same')(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv4_2',border_mode = 'same')(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv4_3',border_mode = 'same')(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv4_4',border_mode = 'same')(x)
    x = MaxPooling2D((2,2), strides=(2,2), dim_ordering="th")(x)

    x = Convolution2D(512, 3, 3, activation='relu',name='conv5_1',border_mode = 'same')(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv5_2',border_mode = 'same')(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv5_3',border_mode = 'same')(x)
    x = Convolution2D(512, 3, 3, activation='relu',name='conv5_4',border_mode = 'same')(x)

    conv_output = Convolution2D(1024, 3, 3, activation='relu',name='conv6_1',border_mode = 'same')(x)

    x = GlobalAveragePooling2D()(conv_output)

    main_output = Dense(2, activation = 'softmax', name="main_output")(x)

    if heatmap:
        model = Model(input=inputs, output=[main_output,conv_output])
    else:
        model = Model(input=inputs, output=main_output)#[main_output,aux_output])

    if weights_path:
        model.load_weights(weights_path,by_name=True)

    return model

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


    conv_output = out[1][0,:,:,:]
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
    cv2.imwrite(output_path, temp)

if __name__ == "__main__":
    
    model = VGG_19_GAP_functional(weights_path='aesthestic_gap_weights_1.h5',heatmap=True)

    delta = 0.0
    store = HDFStore('../dataset_h5/labels.h5','r')
    # delta = 1
    ava_table = store['labels_train']

    ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]
    # X_train = np.hstack(X).reshape(10000,224,224,3)
    # X = pickle.load( open("images_224.p", "rb"))
    h5f = h5py.File('../dataset_h5/images_224_delta_{0}.h5'.format(delta),'r')
    X_train = h5f['data_train']
    #X_train = np.hstack(X).reshape(3,224,224,16160).T

    #X_train = X_train.astype('float32')

    Y_train = ava_table.ix[:, "good"].as_matrix()
    Y_train = to_categorical(Y_train, 2)

    X_test = h5f['data_test']
    ava_test = store['labels_test']
    Y_test = ava_test.ix[:, "good"].as_matrix()
    Y_test = to_categorical(Y_test, 2)

    sgd = SGD(lr=0.001, decay=5e-4, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,loss='categorical_crossentropy', metrics=['accuracy'])

    csv_logger = CSVLogger('training_gap_binary.log')

    #model.fit(X_train,Y_train,
    #    nb_epoch=20, batch_size=32, shuffle="batch", validation_data=(X_test, Y_test), callbacks=[csv_logger])


    ava_path = "../dataset/AVA/data/"



    # for index in ava_test.iloc[::-1][:25].index:
    #     image_name = str(index) + ".jpg"
    #     input_path = ava_path + image_name
    #     output_path = "output/" + image_name
    #     read_and_generate_heatmap(input_path, output_path)


    # saliency_benchmark_dir = "vanishingpoint/"
    # output_dir = "vanishingpoint/output/"
    # for file in os.listdir(saliency_benchmark_dir):
    #     if 'jpg' in file:
    #         read_and_generate_heatmap(saliency_benchmark_dir + file, output_dir + file)


    tag = 17
    semantic_tag_df = ava_test.ix[(ava_test.ix[:,10] == tag) | (ava_test.ix[:,11] == tag)]
    for index in semantic_tag_df.iloc[::-1][:25].index:
        image_name = str(index) + ".jpg"
        input_path = ava_path + image_name
        output_path = "output-semantic/6/" + image_name
        read_and_generate_heatmap(input_path, output_path)

    # style = pd.read_table('../dataset/AVA/style_image_lists/train.jpgl', index_col=0)
    # tag = pd.read_table('../dataset/AVA/style_image_lists/train.lab')

    # style.loc[:,'style'] = tag.as_matrix()

    # ava_with_style = style.join(ava_table, how='inner')

    # vanishing_point = ava_with_style.ix[(ava_with_style.ix[:,'style'] == 13)]

    # vanishing_point = vanishing_point.sort_values(by="score")

    # for index in vanishing_point.iloc[::-1][:25].index:
    #     image_name = str(index) + ".jpg"
    #     input_path = ava_path + image_name
    #     output_path = "output-semantic/6/" + image_name
    #     read_and_generate_heatmap(input_path, output_path)



