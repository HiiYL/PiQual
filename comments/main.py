from keras.layers import Input, Embedding, LSTM, Dense, Activation, GRU,Convolution1D,Dropout
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten
from keras.layers import merge, Reshape, Activation
from keras.layers.pooling import GlobalAveragePooling1D,GlobalMaxPooling1D, MaxPooling1D

from keras.models import Model, Sequential

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD

from keras.regularizers import l2, activity_l2

from keras.callbacks import CSVLogger, ReduceLROnPlateau
from keras.callbacks import CSVLogger, ReduceLROnPlateau,ModelCheckpoint

from pandas import HDFStore
import pandas as pd
import numpy as np
import os

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

GLOVE_DIR = "comments/glove/"


  
def getDistribution(dataframe):
  ratings_matrix = dataframe.ix[:,:10]
  sum_of_ratings = (dataframe.ix[:,:10]).sum(axis=1)
  normalized_score_distribution = ratings_matrix.div(sum_of_ratings,axis='index')
  return normalized_score_distribution.as_matrix()

def getBinaryDistribution(dataframe):
  ratings_matrix = pd.concat([dataframe.ix[:,:5].sum(axis=1),dataframe.ix[:,5:10].sum(axis=1)], axis=1)
  # ratings_matrix = dataframe.ix[:,:10]
  sum_of_ratings = (dataframe.ix[:,:10]).sum(axis=1)
  normalized_score_distribution = ratings_matrix.div(sum_of_ratings,axis='index')
  return normalized_score_distribution.as_matrix()


if __name__ == "__main__":
  store = HDFStore('../dataset/labels.h5')

  ava_table = store['labels_train']
  # ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]

  ava_table = ava_table.sort_values(by="score")
  comments_train = ava_table.ix[:,'comments'].as_matrix()


  Y_train = getDistribution(ava_table)



  ava_test = store['labels_test']
  comments_test = ava_test.ix[:,'comments'].as_matrix()

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
  preds = Dense(10, activation='softmax')(x)

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
  model.compile(loss='kld',
                optimizer='rmsprop',
                metrics=['accuracy'])


  time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  checkpointer = ModelCheckpoint(filepath="text_distribution_weights{}.h5".format(time_now), verbose=1, save_best_only=True)
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=2)
  csv_logger = CSVLogger('training_distribution_text{}.log'.format(time_now))

  model.fit(X_train, Y_train,
            batch_size=128,
            nb_epoch=20,
            validation_data=(X_test, Y_test)
            , callbacks=[checkpointer, reduce_lr, csv_logger])


  out = model.predict(X_test)
  weights = np.array([1,2,3,4,5,6,7,8,9,10])
  score = (out * weights).sum(axis=1)

  good = [ 1 if row >= 5 else 0 for row in score]


  truth_good = ava_test.ix[:, "good"].as_matrix()

  np.sum(good == truth_good) / len(good)







  good = np.argmax(out,axis=1)
  good_truth = ava_test.ix[:, "good"].as_matrix()

  np.sum(good == truth_good) / len(good)