from scipy import ndimage, misc
import cv2
import numpy as np
import os
import pickle
import pandas as pd
from pandas import HDFStore, DataFrame

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding
from keras.utils.np_utils import to_categorical

GLOVE_DIR='comments/glove'

def process_image(image):
    im = np.copy(image)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im

def deprocess_image(image):
    im = np.copy(image)
    im[:,:,0] += 103.939
    im[:,:,1] += 116.779
    im[:,:,2] += 123.

    im = im.transpose((1,2,0))

    return im

def tokenizeAndGenerateIndex(train, test, maxFeatures, maxLength):
  merged = np.concatenate([train, test])
  tokenizer = Tokenizer(nb_words=maxFeatures)
  tokenizer.fit_on_texts(merged)
  sequences_train = tokenizer.texts_to_sequences(train)
  sequences_test = tokenizer.texts_to_sequences(test)
  word_index = tokenizer.word_index
  print('Found %s unique tokens.' % len(word_index))
  data_train = pad_sequences(sequences_train, maxlen=maxLength)
  data_test = pad_sequences(sequences_test, maxlen=maxLength)
  return data_train, data_test, word_index


def generateIndexMappingToEmbedding(embeddingDim):
  embeddings_index = {}
  f = open(os.path.join(GLOVE_DIR, 'glove.6B.{}d.txt'.format(embeddingDim)))
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
  f.close()

  print('Found %s word vectors.' % len(embeddings_index))
  return embeddings_index

def read_and_generate_heatmap(input_path, output_path):
    original_img = cv2.imread(input_path).astype(np.float32)

    width, height, _ = original_img.shape

    im = process_image(cv2.resize(original_img,(224,224)))
    out = model.predict(im)

    class_weights = model.layers[-1].get_weights()[0]
    print("predictions", out[0])

    conv_output = out[2][0,:,:,:]
    #Create the class activation map.
    cam = np.zeros(dtype = np.float32, shape = conv_output.shape[1:3])

    class_to_visualize = 1 # 0 for bad, 1 for good
    for i, w in enumerate(class_weights[:, class_to_visualize]):
            cam += w * conv_output[i, :, :]

    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    temp = heatmap*0.5 + original_img
    file_name = input_path.split("/")[-1].split(".jpg")[0]

    cv2.imwrite('output/' + input_path.split("/")[-1],original_img)
    cv2.imwrite('output/{}_aesthetics.jpg'.format(file_name), temp)


    top_3_hits = np.argsort(out[1], axis=1)[0][::-1][:3]
    print("predictions", top_3_hits)
    semantic_tags = [ semantics.ix[hit + 1].semantic[1:] for hit in top_3_hits] ##NOTE: Quick fix to remove space 


    conv_output = out[3][0,:,:,:]
    #Create the class activation map.
    nth_top_semantic = 0

    for nth_top_semantic in range(len(top_3_hits)):
        cam = np.zeros(dtype = np.float32, shape = conv_output.shape[1:3])
        class_to_visualize = top_3_hits[nth_top_semantic] # 0 for bad, 1 for good
        for i, w in enumerate(class_weights[:, class_to_visualize]):
                cam += w * conv_output[i, :, :]
                
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0
        temp = heatmap*0.5 + original_img
        file_name = input_path.split("/")[-1].split(".jpg")[0]
        cv2.imwrite('output/{}_{}.jpg'.format(file_name, semantic_tags[nth_top_semantic]), temp)


def generate_images_heatmap_with_style(model, ava_table, style_category=-1):
    input_images_dir = "images/"
    output_dir = "output/"

    semantics = pd.read_table('datasetdataset/AVA/tags.txt',delimiter="(\d+)",
     usecols=[1,2], index_col=0, header=None,names=['index','semantic'])

    if style_category != -1:
        style = pd.read_table('datasetdataset/AVA/style_image_lists/train.jpgl', index_col=0)
        tag = pd.read_table('datasetdataset/AVA/style_image_lists/train.lab')

        style.loc[:,'style'] = tag.as_matrix()
        ava_with_style = style.join(ava_table, how='inner')

        images_with_style = ava_with_style.ix[(ava_with_style.ix[:,'style'] == style_category)]
        images_with_style = images_with_style.sort_values(by="score")

        ava_path = "datasetdataset/AVA/data/"
        for index in images_with_style.iloc[::-1][:25].index:
            image_name = str(index) + ".jpg"
            input_path = ava_path + image_name
            output_path = output_dir + image_name
            read_and_generate_heatmap(input_path, output_path)
    else:
        for file in os.listdir(input_images_dir):
            if file.endswith('.jpg'):
                read_and_generate_heatmap(input_images_dir + file, output_dir + file)


def get_distribution(dataframe):
    ratings_matrix = dataframe.ix[:,:10]
    sum_of_ratings = (dataframe.ix[:,:10]).sum(axis=1)
    normalized_score_distribution = ratings_matrix.div(sum_of_ratings,axis='index')
    return normalized_score_distribution.as_matrix()


def prepare_data(delta=0.0,maxFeatures=300, maxEmbeddingInputLength=100, embeddingDim=300, imageDataAvailable=True,
  use_distribution=False, use_semantics=False, use_comments=False):
    store = HDFStore('dataset/labels.h5','r')
    ava_table = store['labels_train']
    ava_test = store['labels_test']

    if delta > 0.0:
        ava_table = ava_table[( abs(ava_table.score - 5) >= delta)]

    if imageDataAvailable:
        h5f = h5py.File('dataset/images_224_delta_{0}.h5'.format(delta),'r')
        X_train = h5f['data_train']
        X_test = h5f['data_test']
    else:
        X_train = []
        X_test = []

    if use_comments:
        comments_train = ava_table.ix[:,'comments'].as_matrix()
        comments_test = ava_test.ix[:,'comments'].as_matrix()

        X_train_text, X_test_text, word_index = tokenizeAndGenerateIndex(
            comments_train, comments_test,
            maxFeatures=maxFeatures, maxLength=maxEmbeddingInputLength)

        embeddings_index = generateIndexMappingToEmbedding(embeddingDim=embeddingDim)
        embedding_matrix = np.zeros((len(word_index) + 1, embeddingDim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(len(word_index) + 1,
                                    embeddingDim,
                                    weights=[embedding_matrix],
                                    input_length=maxEmbeddingInputLength,
                                    trainable=False)

    if use_distribution:
        Y_train = get_distribution(ava_table)
        Y_test = get_distribution(ava_test)
    else:
        Y_train = ava_table.ix[:, "good"].as_matrix()
        Y_train = to_categorical(Y_train, 2)

        Y_test = ava_test.ix[:, "good"].as_matrix()
        Y_test = to_categorical(Y_test, 2)

    if use_semantics:
        Y_train_semantics = to_categorical(ava_table.ix[:,10:12].as_matrix())[:,1:]
        Y_test_semantics = to_categorical(ava_test.ix[:,10:12].as_matrix())[:,1:]

        if use_comments:
            return X_train, Y_train, Y_train_semantics, X_test, Y_test, Y_test_semantics, X_train_text, X_test_text, embedding_layer
        else:
            return X_train, Y_train, Y_train_semantics, X_test, Y_test, Y_test_semantics


    if use_comments:
        return X_train, Y_train, X_test, Y_test, X_train_text, X_test_text, embedding_layer
    else:
        return X_train, Y_train, X_test, Y_test


def evaluate_distribution_accuracy(model, X_test, Y_test):
    y_pred = model.predict(X_test)

    if(isinstance(y_pred,list)):
        aesthetics_pred = np.array(y_pred[0])
    else:
        aesthetics_pred = np.array(y_pred)


    if(isinstance(Y_test,list)):
        aesthetics_label = np.array(Y_test[0])
    else:
        aesthetics_label = np.array(Y_test)


    weights = np.array([1,2,3,4,5,6,7,8,9,10])

    score_pred = (aesthetics_pred * weights).sum(axis=1)
    score_test = (aesthetics_label * weights).sum(axis=1)

    Y_pred_binary = np.array([ 1 if row >= 5 else 0 for row in score_pred])
    Y_test_binary = np.array([ 1 if row >= 5 else 0 for row in score_test])


    accuracy = np.sum(Y_pred_binary == Y_test_binary) / len(Y_test_binary)

    print("accuracy = {} %".format(accuracy * 100))
    return accuracy