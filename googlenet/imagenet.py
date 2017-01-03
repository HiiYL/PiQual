from scipy.misc import imread, imresize

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from googlenet_custom_layers import PoolHelper,LRN


def create_googlenet(weights_path=None):
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
    
    
    loss1_ave_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),name='loss1/ave_pool')(inception_4a_output)
    
    loss1_conv = Convolution2D(128,1,1,border_mode='same',activation='relu',name='loss1/conv',W_regularizer=l2(0.0002))(loss1_ave_pool)
    
    loss1_flat = Flatten()(loss1_conv)
    
    loss1_fc = Dense(1024,activation='relu',name='loss1/fc',W_regularizer=l2(0.0002))(loss1_flat)
    
    loss1_drop_fc = Dropout(0.7)(loss1_fc)
    
    loss1_classifier = Dense(1000,name='loss1/classifier',W_regularizer=l2(0.0002))(loss1_drop_fc)
    
    loss1_classifier_act = Activation('softmax')(loss1_classifier)
    
    
    inception_4b_1x1 = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_4b/1x1',W_regularizer=l2(0.0002))(inception_4a_output)
    
    inception_4b_3x3_reduce = Convolution2D(112,1,1,border_mode='same',activation='relu',name='inception_4b/3x3_reduce',W_regularizer=l2(0.0002))(inception_4a_output)
    
    inception_4b_3x3 = Convolution2D(224,3,3,border_mode='same',activation='relu',name='inception_4b/3x3',W_regularizer=l2(0.0002))(inception_4b_3x3_reduce)
    
    inception_4b_5x5_reduce = Convolution2D(24,1,1,border_mode='same',activation='relu',name='inception_4b/5x5_reduce',W_regularizer=l2(0.0002))(inception_4a_output)
    
    inception_4b_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4b/5x5',W_regularizer=l2(0.0002))(inception_4b_5x5_reduce)
    
    inception_4b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4b/pool')(inception_4a_output)
    
    inception_4b_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4b/pool_proj',W_regularizer=l2(0.0002))(inception_4b_pool)
    
    inception_4b_output = merge([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj],mode='concat',concat_axis=1,name='inception_4b_output')
    
    
    inception_4c_1x1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4c/1x1',W_regularizer=l2(0.0002))(inception_4b_output)
    
    inception_4c_3x3_reduce = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4c/3x3_reduce',W_regularizer=l2(0.0002))(inception_4b_output)
    
    inception_4c_3x3 = Convolution2D(256,3,3,border_mode='same',activation='relu',name='inception_4c/3x3',W_regularizer=l2(0.0002))(inception_4c_3x3_reduce)
    
    inception_4c_5x5_reduce = Convolution2D(24,1,1,border_mode='same',activation='relu',name='inception_4c/5x5_reduce',W_regularizer=l2(0.0002))(inception_4b_output)
    
    inception_4c_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4c/5x5',W_regularizer=l2(0.0002))(inception_4c_5x5_reduce)
    
    inception_4c_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4c/pool')(inception_4b_output)
    
    inception_4c_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4c/pool_proj',W_regularizer=l2(0.0002))(inception_4c_pool)
    
    inception_4c_output = merge([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj],mode='concat',concat_axis=1,name='inception_4c/output')
    
    
    inception_4d_1x1 = Convolution2D(112,1,1,border_mode='same',activation='relu',name='inception_4d/1x1',W_regularizer=l2(0.0002))(inception_4c_output)
    
    inception_4d_3x3_reduce = Convolution2D(144,1,1,border_mode='same',activation='relu',name='inception_4d/3x3_reduce',W_regularizer=l2(0.0002))(inception_4c_output)
    
    inception_4d_3x3 = Convolution2D(288,3,3,border_mode='same',activation='relu',name='inception_4d/3x3',W_regularizer=l2(0.0002))(inception_4d_3x3_reduce)
    
    inception_4d_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_4d/5x5_reduce',W_regularizer=l2(0.0002))(inception_4c_output)
    
    inception_4d_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4d/5x5',W_regularizer=l2(0.0002))(inception_4d_5x5_reduce)
    
    inception_4d_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4d/pool')(inception_4c_output)
    
    inception_4d_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4d/pool_proj',W_regularizer=l2(0.0002))(inception_4d_pool)
    
    inception_4d_output = merge([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj],mode='concat',concat_axis=1,name='inception_4d/output')
    
    
    loss2_ave_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),name='loss2/ave_pool')(inception_4d_output)
    
    loss2_conv = Convolution2D(128,1,1,border_mode='same',activation='relu',name='loss2/conv',W_regularizer=l2(0.0002))(loss2_ave_pool)
    
    loss2_flat = Flatten()(loss2_conv)
    
    loss2_fc = Dense(1024,activation='relu',name='loss2/fc',W_regularizer=l2(0.0002))(loss2_flat)
    
    loss2_drop_fc = Dropout(0.7)(loss2_fc)
    
    loss2_classifier = Dense(1000,name='loss2/classifier',W_regularizer=l2(0.0002))(loss2_drop_fc)
    
    loss2_classifier_act = Activation('softmax')(loss2_classifier)
    
    
    inception_4e_1x1 = Convolution2D(256,1,1,border_mode='same',activation='relu',name='inception_4e/1x1',W_regularizer=l2(0.0002))(inception_4d_output)
    
    inception_4e_3x3_reduce = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_4e/3x3_reduce',W_regularizer=l2(0.0002))(inception_4d_output)
    
    inception_4e_3x3 = Convolution2D(320,3,3,border_mode='same',activation='relu',name='inception_4e/3x3',W_regularizer=l2(0.0002))(inception_4e_3x3_reduce)
    
    inception_4e_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_4e/5x5_reduce',W_regularizer=l2(0.0002))(inception_4d_output)
    
    inception_4e_5x5 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='inception_4e/5x5',W_regularizer=l2(0.0002))(inception_4e_5x5_reduce)
    
    inception_4e_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4e/pool')(inception_4d_output)
    
    inception_4e_pool_proj = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4e/pool_proj',W_regularizer=l2(0.0002))(inception_4e_pool)
    
    inception_4e_output = merge([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj],mode='concat',concat_axis=1,name='inception_4e/output')
    
    
    inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)
    
    pool4_helper = PoolHelper()(inception_4e_output_zero_pad)
    
    pool4_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool4/3x3_s2')(pool4_helper)
    
    
    inception_5a_1x1 = Convolution2D(256,1,1,border_mode='same',activation='relu',name='inception_5a/1x1',W_regularizer=l2(0.0002))(pool4_3x3_s2)
    
    inception_5a_3x3_reduce = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_5a/3x3_reduce',W_regularizer=l2(0.0002))(pool4_3x3_s2)
    
    inception_5a_3x3 = Convolution2D(320,3,3,border_mode='same',activation='relu',name='inception_5a/3x3',W_regularizer=l2(0.0002))(inception_5a_3x3_reduce)
    
    inception_5a_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_5a/5x5_reduce',W_regularizer=l2(0.0002))(pool4_3x3_s2)
    
    inception_5a_5x5 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='inception_5a/5x5',W_regularizer=l2(0.0002))(inception_5a_5x5_reduce)
    
    inception_5a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_5a/pool')(pool4_3x3_s2)
    
    inception_5a_pool_proj = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_5a/pool_proj',W_regularizer=l2(0.0002))(inception_5a_pool)
    
    inception_5a_output = merge([inception_5a_1x1,inception_5a_3x3,inception_5a_5x5,inception_5a_pool_proj],mode='concat',concat_axis=1,name='inception_5a/output')
    
    
    inception_5b_1x1 = Convolution2D(384,1,1,border_mode='same',activation='relu',name='inception_5b/1x1',W_regularizer=l2(0.0002))(inception_5a_output)
    
    inception_5b_3x3_reduce = Convolution2D(192,1,1,border_mode='same',activation='relu',name='inception_5b/3x3_reduce',W_regularizer=l2(0.0002))(inception_5a_output)
    
    inception_5b_3x3 = Convolution2D(384,3,3,border_mode='same',activation='relu',name='inception_5b/3x3',W_regularizer=l2(0.0002))(inception_5b_3x3_reduce)
    
    inception_5b_5x5_reduce = Convolution2D(48,1,1,border_mode='same',activation='relu',name='inception_5b/5x5_reduce',W_regularizer=l2(0.0002))(inception_5a_output)
    
    inception_5b_5x5 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='inception_5b/5x5',W_regularizer=l2(0.0002))(inception_5b_5x5_reduce)
    
    inception_5b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_5b/pool')(inception_5a_output)
    
    inception_5b_pool_proj = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_5b/pool_proj',W_regularizer=l2(0.0002))(inception_5b_pool)
    
    inception_5b_output = merge([inception_5b_1x1,inception_5b_3x3,inception_5b_5x5,inception_5b_pool_proj],mode='concat',concat_axis=1,name='inception_5b/output')
    
    
    pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7),strides=(1,1),name='pool5/7x7_s2')(inception_5b_output)
    
    loss3_flat = Flatten()(pool5_7x7_s1)
    
    pool5_drop_7x7_s1 = Dropout(0.4)(loss3_flat)
    
    loss3_classifier = Dense(1000,name='loss3/classifier',W_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
    
    loss3_classifier_act = Activation('softmax',name='prob')(loss3_classifier)
    
    
    googlenet = Model(input=input, output=[loss3_flat])
    
    if weights_path:
        googlenet.load_weights(weights_path)
    
    return googlenet




# model = create_googlenet('weights/googlenet_weights.h5')
