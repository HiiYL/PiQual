from scipy import misc
import copy
import numpy as np
from squeezenet import get_squeezenet
import time
from pandas import HDFStore, DataFrame
import pandas as pd
import h5py

from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD

from keras.callbacks import CSVLogger


if __name__ == '__main__':
    delta = 0.0
    store = HDFStore('../dataset_h5/labels.h5')
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

    model = get_squeezenet(2, dim_ordering='tf')
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

    csv_logger = CSVLogger('training_squeezenet_lr_0.001.log')
    model.fit(X_train, Y_train, nb_epoch=10,shuffle="batch",batch_size=16,validation_data=(X_test, Y_test), callbacks=[csv_logger])
    # model.load_weights('model/squeezenet_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)

    start = time.time()
    im = misc.imread('asian_dad.jpg')

    im = misc.imresize(im, (224, 224)).astype(np.float32)
    aux = copy.copy(im)
    im[:, :, 0] = aux[:, :, 2]
    im[:, :, 2] = aux[:, :, 0]

    # Remove image mean
    im[:, :, 0] -= 104.006
    im[:, :, 1] -= 116.669
    im[:, :, 2] -= 122.679

    #im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)

    res = model.predict(im)
    classes = []
    with open('classes.txt', 'r') as list_:
        for line in list_:
            classes.append(line.rstrip('\n'))
    duration = time.time() - start
    print("{} s to get output".format(duration))

    print('class: ' + classes[np.argmax(res[0])] + ' acc: ' + str(res[0][np.argmax(res[0])]))
