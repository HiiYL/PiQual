from keras.layers import Input, Embedding, LSTM, Dense, Activation, GRU
from keras.models import Model


from pandas import HDFStore
import pandas as pd
import numpy as np
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical

MAX_NB_WORDS=100
MAX_SEQUENCE_LENGTH=100

def tokenizeAndGenerateIndex(texts):
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return data


question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = GRU(256)(embedded_question)
output = Dense(2, activation='softmax')(encoded_question)

model = Model(input=question_input, output=output)

store = HDFStore('dataset_h5/labels.h5')


ava_table = store['labels_train']
comments_train = ava_table.ix[:,'comments'].as_matrix()
X_train = tokenizeAndGenerateIndex(comments_train)

Y_train = ava_table.ix[:, "good"].as_matrix()
Y_train = to_categorical(Y_train, 2)



ava_test = store['labels_test']
comments_test = ava_test.ix[:,'comments'].as_matrix()
X_test = tokenizeAndGenerateIndex(comments_test)

Y_test = ava_test.ix[:, "good"].as_matrix()
Y_test = to_categorical(Y_test, 2)


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(X_train, Y_train,validation_data=(X_test,Y_test),nb_epoch=2, batch_size=128)