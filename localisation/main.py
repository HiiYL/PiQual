from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Input, Dense, Activation
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.models import Model
from keras import backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D

import cv2

import matplotlib.pyplot as plt
import numpy as np

def VGG_19_GAP_functional(weights_path=None,heatmap=False):

    inputs = Input(shape=(3, None, None))

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

    x = ZeroPadding2D((1,1))(x)
    final_conv = Convolution2D(1024, 3, 3, activation='relu',name='conv6_1')(x)

    x = GlobalAveragePooling2D()(final_conv)

    main_output = Dense(2, activation = 'softmax', init='uniform', name="main_output")(x)
    aux_output = final_conv

    model = Model(input=inputs, output=[main_output,aux_output])

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

    original_img = cv2.imread('kitten.jpg').astype(np.float32)

    width, height, _ = original_img.shape

    im = np.copy(original_img)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model = VGG_19_GAP_functional(weights_path='../binary_cnn_10_named_weights.h5')



    model.compile(optimizer=sgd, loss='mse')

    out = model.predict(im)

    print(out.shape)

    output_path = "output.jpg"


    #Get the 512 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    # final_conv_layer = get_output_layer(model, "conv5_3")
    # get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    # [conv_outputs, predictions] = get_output([im])
    # conv_outputs = conv_outputs[0, :, :, :]

    out[1] = out[1][0,:,:,:]

    #Create the class activation map.
    cam = np.zeros(dtype = np.float32, shape = out[1].shape[1:3])
    for i, w in enumerate(class_weights[:, 1]):
            cam += w * out[1][i, :, :]
    print("predictions", out[0])
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    im = heatmap*0.5 + original_img
    cv2.imwrite(output_path, im)
    # heatmap = out[0,:,:].sum(axis=0)

    # im = plt.imread('kitten.jpg')
    # implot = plt.imshow(im)


    # plt.imshow(heatmap, cmap='hot', interpolation='bilinear',alpha=0.7,aspect='auto',extent=[0,im.shape[1],im.shape[0],0])
    # plt.show()