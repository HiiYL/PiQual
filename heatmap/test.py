from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D

import cv2

import matplotlib.pyplot as plt
import numpy as np

def VGG_19(weights_path=None,heatmap=False):
    model = Sequential()

    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    #model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    #model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_4'))
    #model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_4'))
    #model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_4'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(1024, 3, 3, activation='relu',name='conv6_1'))

    if weights_path:
        model.load_weights(weights_path,by_name=True)

    return model

if __name__ == "__main__":
    # ### Here is a script to compute the heatmap of the dog synsets.
    # ## We find the synsets corresponding to dogs on ImageNet website
    # s = "n02084071"
    # ids = synset_to_dfs_ids(s)
    # # Most of the synsets are not in the subset of the synsets used in ImageNet recognition task.
    # ids = np.array([id_ for id_ in ids if id_ is not None])

    # im = preprocess_image_batch(['examples/dog.jpg'], color_mode="rgb")

    # Test pretrained model

    im = cv2.resize(cv2.imread('tazeek-punch.jpg'), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model = VGG_19(weights_path='../binary_cnn_10_named_weights.h5')
    model.compile(optimizer=sgd, loss='mse')

    out = model.predict(im)
    heatmap = out[0,:,:].sum(axis=0)

    im = plt.imread('tazeek-punch.jpg')
    implot = plt.imshow(im)


    plt.imshow(heatmap, cmap='hot', interpolation='bilinear',alpha=0.7,aspect='auto',extent=[0,im.shape[1],im.shape[0],0])
    plt.show()