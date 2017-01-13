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

from keras.callbacks import CSVLogger, ReduceLROnPlateau,ModelCheckpoint
from keras.layers.pooling import GlobalAveragePooling1D,GlobalMaxPooling1D, MaxPooling1D

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation


from keras.regularizers import l2, activity_l2


from datetime import datetime

max_features = 20000
maxlen=100
batch_size = 64

hidden_dims = 250
nb_epoch = 100

EMBEDDING_DIM = 300
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
  f = open(os.path.join(GLOVE_DIR, 'glove.6B.{}d.txt'.format(EMBEDDING_DIM)))
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
  f.close()

  print('Found %s word vectors.' % len(embeddings_index))
  return embeddings_index
  

def getDistribution(dataframe):
    ratings_matrix = dataframe.ix[:,:10]
    sum_of_ratings = (dataframe.ix[:,:10]).sum(axis=1)
    normalized_score_distribution = ratings_matrix.div(sum_of_ratings,axis='index')
    return normalized_score_distribution.as_matrix()



def evaluate_distribution_accuracy(model, X_test, Y_test):
    y_pred = model.predict(X_test)

    if(y_pred is list):
        aesthetics_pred = y_pred[0]
    else:
        aesthetics_pred = y_pred


    weights = np.array([1,2,3,4,5,6,7,8,9,10])

    score_pred = (aesthetics_pred * weights).sum(axis=1)
    score_test = (Y_test * weights).sum(axis=1)

    Y_pred_binary = np.array([ 1 if row >= 5 else 0 for row in score_pred])
    Y_test_binary = np.array([ 1 if row >= 5 else 0 for row in score_test])


    accuracy = np.sum(Y_pred_binary == Y_test_binary) / len(Y_test_binary)

    print("accuracy = {} %".format(accuracy * 100))
    return accuracy


store = HDFStore('../dataset_h5/labels.h5')

ava_table = store['labels_train']
# ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]

ava_table = ava_table.sort_values(by="score")
comments_train = ava_table.ix[:,'comments'].as_matrix()


# Y_train = ava_table.ix[:, "good"].as_matrix()
# Y_train = to_categorical(Y_train, 2)

Y_train = getDistribution(ava_table)


ava_test = store['labels_test']
comments_test = ava_test.ix[:,'comments'].as_matrix()

# Y_test = ava_test.ix[:, "good"].as_matrix()
# Y_test = to_categorical(Y_test, 2)

Y_test = getDistribution(ava_test)

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
                            trainable=False, name="embedding")

# comment_input = Input(shape=(100,), dtype='int32')
# embedded_sequences = embedding_layer(comment_input)
# # x = GRU(EMBEDDING_DIM)(embedded_sequences) # 0.8013
# # x = GRU(EMBEDDING_DIM,dropout_W = 0.3,dropout_U = 0.3)(embedded_sequences) #0.8109


# # x = Convolution1D(128, 5, activation='relu')(embedded_sequences)
# # x = MaxPooling1D(5)(x)
# # x = Convolution1D(128, 5, activation='relu')(x)
# # x = MaxPooling1D(5)(x)
# # x = Convolution1D(128, 5, activation='relu')(x)
# # x = MaxPooling1D(35)(x)  # global max pooling
# # x = GRU(EMBEDDING_DIM,dropout_W = 0.3,dropout_U = 0.3,return_sequences=True, name="gru")(embedded_sequences)
# # # x = GRU(EMBEDDING_DIM,dropout_W = 0.3,dropout_U = 0.3,return_sequences=True, name="gru_2")(x)
# # # x = GRU(EMBEDDING_DIM,dropout_W = 0.3,dropout_U = 0.3,return_sequences=True, name="gru_3")(x)
# # x = GRU(EMBEDDING_DIM,dropout_W = 0.3,dropout_U = 0.3, name="gru_2")(x)
# # x = Flatten()(x)
# # x = Dense(128, activation='relu')(x)
# # x = Dropout(0.5)(x)

# # x = Flatten()(embedded_sequences)

# x = LSTM(EMBEDDING_DIM,batch_input_shape=(128, 1, 1),dropout_W = 0.3,dropout_U = 0.3,return_sequences=True, name="lstm",stateful=True)(embedded_sequences)
# # x = LSTM(EMBEDDING_DIM,dropout_W = 0.3,dropout_U = 0.3,return_sequences=True, name="lstm_2")(x)
# # x = GRU(EMBEDDING_DIM,dropout_W = 0.3,dropout_U = 0.3,return_sequences=True, name="gru_3")(x)
# x = LSTM(EMBEDDING_DIM,batch_input_shape=(128, 1, 1),dropout_W = 0.3,dropout_U = 0.3, name="lstm_2",stateful=True)(x)



# preds = Dense(10, name="comment_output",activation='softmax')(x)

# # question_input = Input(shape=(maxlen,), dtype='int32')
# # x = Embedding(input_dim=max_features,
# #  output_dim=EMBEDDING_DIM, input_length=maxlen,
# #    dropout=0.25)(question_input)
# # # x = Convolution1D(nb_filter=nb_filter,
# # #                         filter_length=filter_length,
# # #                         border_mode='valid',
# # #                         activation='relu',
# # #                         subsample_length=1)(x)
# # # x = MaxPooling1D(pool_length=pool_length)(x)
# # x = GRU(EMBEDDING_DIM,dropout_W = 0.3,dropout_U = 0.3)(x)
# # output = Dense(2, activation='softmax')(x)

# model = Model(input=comment_input, output=preds)

# model.load_weights('text_binary_weights2016-12-07 09:31:43.h5')

# model.save_weights('text_binary_weights_named.h5')
# sgd = SGD(lr=0.01, decay=5e-4, momentum=0.9, nesterov=True, clipnorm=1., clipvalue=0.5)




model = Sequential()
model.add(LSTM(EMBEDDING_DIM, batch_input_shape=(128, 100, 1), return_sequences=True, stateful=True))
model.add(LSTM(EMBEDDING_DIM, batch_input_shape=(128, 100, 1), return_sequences=False, stateful=True))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_identifier = "text_stateful_distribution"
# time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
checkpointer = ModelCheckpoint(filepath="weights/{} - {}.h5".format(time_now, model_identifier), verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=2)
csv_logger = CSVLogger('logs/{} - {}.log'.format(time_now,model_identifier))


# model.load_weights("weights/{} - {}.h5".format("2017-01-06 00:03:26",model_identifier))

model.compile(loss='kld',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(X_train, Y_train,
          batch_size=128,
          nb_epoch=10,
          validation_data=(X_test, Y_test), callbacks=[checkpointer, reduce_lr, csv_logger])


accuracy = evaluate_distribution_accuracy(model, X_test, Y_test)