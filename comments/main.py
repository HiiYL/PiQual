from keras.layers import Input, Embedding, LSTM, Dense, Activation, GRU,Convolution1D,Dropout
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
from keras.layers.pooling import GlobalAveragePooling1D,GlobalMaxPooling1D, MaxPooling1D

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation


from keras.regularizers import l2, activity_l2

max_features = 20000
maxlen=100
batch_size = 64

hidden_dims = 250
nb_epoch = 100

embedding_dims = 128
delta = 1.0

# Convolution
filter_length = 5
nb_filter = 64
pool_length = 4


def tokenizeAndGenerateIndex(texts):
    tokenizer = Tokenizer(nb_words=max_features)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=maxlen)
    return data

store = HDFStore('../dataset_h5/labels.h5')


ava_table = store['labels_train']

# ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]

ava_table = ava_table.sort_values(by="score")
comments_train = ava_table.ix[:,'comments'].as_matrix()
X_train = tokenizeAndGenerateIndex(comments_train)

Y_train = ava_table.ix[:, "good"].as_matrix()
Y_train = to_categorical(Y_train, 2)



ava_test = store['labels_test']
comments_test = ava_test.ix[:,'comments'].as_matrix()
X_test = tokenizeAndGenerateIndex(comments_test)

Y_test = ava_test.ix[:, "good"].as_matrix()
Y_test = to_categorical(Y_test, 2)


question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=max_features,
 output_dim=128, dropout=0.25,
  input_length=100)(question_input)

# embedded_question = Flatten()(embedded_question)
encoded_question = GRU(128,W_regularizer=l2(0.5),U_regularizer=l2(0.1))(embedded_question)

output = Dense(2, activation='softmax')(encoded_question)
# question_input = Input(shape=(maxlen,), dtype='int32')
# x = Embedding(input_dim=max_features,
#  output_dim=embedding_dims, input_length=maxlen,
#    dropout=0.25)(question_input)
# # x = Convolution1D(nb_filter=nb_filter,
# #                         filter_length=filter_length,
# #                         border_mode='valid',
# #                         activation='relu',
# #                         subsample_length=1)(x)
# # x = MaxPooling1D(pool_length=pool_length)(x)
# x = GRU(embedding_dims,dropout_W = 0.3,dropout_U = 0.3)(x)
# output = Dense(2, activation='softmax')(x)

model = Model(input=question_input, output=output)
# sgd = SGD(lr=0.01, decay=5e-4, momentum=0.9, nesterov=True, clipnorm=1., clipvalue=0.5)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test))


### CNN ###


# model = Sequential()

# # we start off with an efficient embedding layer which maps
# # our vocab indices into embedding_dims dimensions
# model.add(Embedding(max_features + 2,
#                     embedding_dims,
#                     input_length=maxlen,
#                     dropout=0.2, mask_zero=True))

# # we add a Convolution1D, which will learn nb_filter
# # word group filters of size filter_length:
# model.add(Convolution1D(nb_filter=nb_filter,
#                         filter_length=filter_length,
#                         border_mode='valid',
#                         activation='relu',
#                         subsample_length=1))
# # we use max pooling:
# model.add(GlobalMaxPooling1D())

# # We add a vanilla hidden layer:
# model.add(Dense(hidden_dims, activation='relu'))
# model.add(Dropout(0.2))

# model.add(Dense(2, activation='softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])


# model.fit(X_train, Y_train,
#           batch_size=batch_size,
#           nb_epoch=nb_epoch,
#           validation_data=(X_test, Y_test))


### GRU ###






# model = Sequential()

# model.add(Embedding(max_features,
#                     embedding_dims,
#                     input_length=maxlen))

# # we add a GlobalAveragePooling1D, which will average the embeddings
# # of all words in the document
# model.add(GlobalAveragePooling1D())

# # We project onto a single unit output layer, and squash it with a sigmoid:
# model.add(Dense(2, activation='softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# model.fit(X_train, Y_train,
#           batch_size=32,
#           nb_epoch=10,
#           validation_data=(X_test, Y_test))