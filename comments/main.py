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
from keras.layers.pooling import GlobalAveragePooling1D,GlobalMaxPooling1D

max_features=20000
maxlen=100
batch_size = 32
embedding_dims = 50
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 20

embedding_dims = 50

delta = 0.0

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

ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]
comments_train = ava_table.ix[:,'comments'].as_matrix()
X_train = tokenizeAndGenerateIndex(comments_train)

Y_train = ava_table.ix[:, "good"].as_matrix()
Y_train = to_categorical(Y_train, 2)



ava_test = store['labels_test']
comments_test = ava_test.ix[:,'comments'].as_matrix()
X_test = tokenizeAndGenerateIndex(comments_test)

Y_test = ava_test.ix[:, "good"].as_matrix()
Y_test = to_categorical(Y_test, 2)


# model = Sequential()

# # we start off with an efficient embedding layer which maps
# # our vocab indices into embedding_dims dimensions
# model.add(Embedding(max_features,
#                     embedding_dims,
#                     input_length=maxlen,
#                     dropout=0.2))

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



question_input = Input(shape=(maxlen,), dtype='int32')
embedded_question = Embedding(input_dim=max_features + 2,
 output_dim=embedding_dims, input_length=maxlen, dropout=0.5,
  mask_zero=True)(question_input)
encoded_question = GRU(embedding_dims,dropout_W = 0.3,dropout_U = 0.3)(embedded_question)
output = Dense(2, activation='softmax')(encoded_question)

model = Model(input=question_input, output=output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test))



# model = Sequential()
# model.add(Embedding(max_features, 128, dropout=0.2))
# model.add(GRU(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
# model.add(Dense(2, activation='softmax'))

# # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=2)

# # sgd = SGD(lr=0.01, decay=5e-4, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['acc'])

# model.fit(X_train, Y_train,validation_data=(X_test,Y_test),nb_epoch=20, batch_size=32)


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