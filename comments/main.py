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

EMBEDDING_DIM = 100
delta = 1.0

# Convolution
filter_length = 5
nb_filter = 64
pool_length = 4


GLOVE_DIR = "glove/"

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
  f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
  f.close()

  print('Found %s word vectors.' % len(embeddings_index))
  return embeddings_index
  


store = HDFStore('../dataset_h5/labels.h5')

ava_table = store['labels_train']
# ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]

ava_table = ava_table.sort_values(by="score")
comments_train = ava_table.ix[:,'comments'].as_matrix()


Y_train = ava_table.ix[:, "good"].as_matrix()
Y_train = to_categorical(Y_train, 2)



ava_test = store['labels_test']
comments_test = ava_test.ix[:,'comments'].as_matrix()

Y_test = ava_test.ix[:, "good"].as_matrix()
Y_test = to_categorical(Y_test, 2)

X_train, X_test, word_index = tokenizeAndGenerateIndex(comments_train, comments_test)

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

comment_input = Input(shape=(100,), dtype='int32')
embedded_sequences = embedding_layer(comment_input)
# x = GRU(EMBEDDING_DIM)(embedded_sequences) # 0.8013
# x = GRU(EMBEDDING_DIM,dropout_W = 0.3,dropout_U = 0.3)(embedded_sequences) #0.8109


# x = Convolution1D(128, 5, activation='relu')(embedded_sequences)
# x = MaxPooling1D(5)(x)
# x = Convolution1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(5)(x)
# x = Convolution1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(35)(x)  # global max pooling
x = GRU(EMBEDDING_DIM,dropout_W = 0.3,dropout_U = 0.3)(embedded_sequences)
# x = Flatten()(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)

# x = Flatten()(embedded_sequences)
preds = Dense(2, activation='softmax')(x)

# question_input = Input(shape=(maxlen,), dtype='int32')
# x = Embedding(input_dim=max_features,
#  output_dim=EMBEDDING_DIM, input_length=maxlen,
#    dropout=0.25)(question_input)
# # x = Convolution1D(nb_filter=nb_filter,
# #                         filter_length=filter_length,
# #                         border_mode='valid',
# #                         activation='relu',
# #                         subsample_length=1)(x)
# # x = MaxPooling1D(pool_length=pool_length)(x)
# x = GRU(EMBEDDING_DIM,dropout_W = 0.3,dropout_U = 0.3)(x)
# output = Dense(2, activation='softmax')(x)

model = Model(input=comment_input, output=preds)
# sgd = SGD(lr=0.01, decay=5e-4, momentum=0.9, nesterov=True, clipnorm=1., clipvalue=0.5)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(X_train, Y_train,
          batch_size=128,
          nb_epoch=20,
          validation_data=(X_test, Y_test))