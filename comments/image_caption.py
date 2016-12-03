from keras.layers import Input, Embedding, LSTM, Dense, Activation, GRU,Convolution1D,Convolution2D,Dropout,Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Flatten, Dense, Dropout, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, Sequential


from pandas import HDFStore
import pandas as pd
import numpy as np
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical


from keras.optimizers import SGD

from keras.callbacks import CSVLogger, ReduceLROnPlateau
from keras.layers.pooling import GlobalAveragePooling1D,GlobalMaxPooling1D

import h5py

max_caption_len = 16
vocab_size = 20000


maxlen=16

delta = 0.0

def tokenizeAndGenerateIndex(texts):
    tokenizer = Tokenizer(nb_words=vocab_size)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=maxlen,padding='post')
    return data


h5f = h5py.File('../dataset_h5/images_224_delta_{0}.h5'.format(delta),'r')
X_train = h5f['data_train']


store = HDFStore('../dataset_h5/labels.h5')
ava_table = store['labels_train']

comments_train = ava_table.ix[:,'comments'].as_matrix()
texts_indices = tokenizeAndGenerateIndex(comments_train)

partial_captions = np.ndarray(shape=(len(X_train),), dtype=int)
for i in enumerate(texts_indices):
	temp = []
	for j in enumerate(texts_indices[i]):

		while texts_indices[i][j+1] != 0:



# inv_map = {v: k for k, v in word_index.items()}



# first, let's define an image model that
# will encode pictures into 128-dimensional vectors.
# it should be initialized with pre-trained weights.
image_model = Sequential()
image_model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 224, 224)))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(32, 3, 3))
image_model.add(Activation('relu'))
image_model.add(MaxPooling2D(pool_size=(2, 2)))

image_model.add(Convolution2D(64, 3, 3, border_mode='valid'))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(64, 3, 3))
image_model.add(Activation('relu'))
image_model.add(MaxPooling2D(pool_size=(2, 2)))

image_model.add(Flatten())
image_model.add(Dense(128))

# let's load the weights from a save file.
# image_model.load_weights('weight_file.h5')

# next, let's define a RNN model that encodes sequences of words
# into sequences of 128-dimensional word vectors.
language_model = Sequential()
language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
language_model.add(GRU(output_dim=128, return_sequences=True))
language_model.add(TimeDistributed(Dense(128)))

# let's repeat the image vector to turn it into a sequence.
image_model.add(RepeatVector(max_caption_len))

# the output of both models will be tensors of shape (samples, max_caption_len, 128).
# let's concatenate these 2 vector sequences.
model = Sequential()
model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
# let's encode this vector sequence into a single vector
model.add(GRU(256, return_sequences=False))
# which will be used to compute a probability
# distribution over what the next word in the caption should be!
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# "images" is a numpy float array of shape (nb_samples, nb_channels=3, width, height).
# "captions" is a numpy integer array of shape (nb_samples, max_caption_len)
# containing word index sequences representing partial captions.
# "next_words" is a numpy float array of shape (nb_samples, vocab_size)
# containing a categorical encoding (0s and 1s) of the next word in the corresponding
# partial caption.
model.fit([images, partial_captions], next_words, batch_size=16, nb_epoch=100)
out = model.predict([X_train[:2], np.random.rand(2,16)])