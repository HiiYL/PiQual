from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Embedding,GRU,TimeDistributedDense,RepeatVector,Merge
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
import cv2
import numpy as np

max_caption_len = 21
vocab_size = 43
def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    #Remove the last two layers to get the 4096D activations
    model.layers.pop()
    model.layers.pop()

    return model
print("VGG loading")
image_model = VGG_16('')
image_model.trainable = False
print("VGG loaded")
# let's load the weights from a save file.
# image_model.load_weights('weight_file.h5')

# next, let's define a RNN model that encodes sequences of words
# into sequences of 128-dimensional word vectors.
print("Text model loading")
language_model = Sequential()
language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
language_model.add(GRU(output_dim=128, return_sequences=True))
language_model.add(TimeDistributedDense(128))
print("Text model loaded")
# let's repeat the image vector to turn it into a sequence.
print("Repeat model loading")
image_model.add(RepeatVector(max_caption_len))
print("Repeat model loaded")
# the output of both models will be tensors of shape (samples, max_caption_len, 128).
# let's concatenate these 2 vector sequences.
print("Merging")
model = Sequential()
model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
# let's encode this vector sequence into a single vector
model.add(GRU(256, return_sequences=False))
# which will be used to compute a probability
# distribution over what the next word in the caption should be!
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print("Merged")
# "images" is a numpy float array of shape (nb_samples, nb_channels=3, width, height).
# "captions" is a numpy integer array of shape (nb_samples, max_caption_len)
# containing word index sequences representing partial captions.
# "next_words" is a numpy float array of shape (nb_samples, vocab_size)
# containing a categorical encoding (0s and 1s) of the next word in the corresponding
# partial caption.
print("Data preprocessig")
Texts = ["START A girl is stretched out in shallow water END",
        "START The two people stand by a body of water and in front of bushes in fall END",
        "START A blonde horse and a blonde girl in a black sweatshirt are staring at a fire in a barrel END",
        "START Children sit and watch the fish moving in the pond END",
        "START A fisherman fishes at the bank of a foggy river END"]

Images = ["667626_18933d713e.jpg",
         "3637013_c675de7705.jpg",
         "10815824_2997e03d76.jpg",
         "12830823_87d2654e31.jpg",
         "17273391_55cfc7d3d4.jpg"]
images = []
for image in Images:
    img = cv2.imread(image)
    img.resize((3,224,224))
    images.append(img)
images = np.asarray(images)

words = [txt.split() for txt in Texts]
unique = []
for word in words:
    unique.extend(word)
unique = list(set(unique))
word_index = {}
index_word = {}
for i,word in enumerate(unique):
    word_index[word] = i
    index_word[i] = word

partial_captions = []
for text in Texts:
    one = [word_index[txt] for txt in text.split()]
    # print(one)
    partial_captions.append(one)

partial_captions = sequence.pad_sequences(partial_captions, maxlen=max_caption_len,padding='post')
next_words = np.zeros((5,vocab_size))
for i,text in enumerate(Texts):
    text = text.split()
    x = [word_index[txt] for txt in text]
    print(x)
    x = np.asarray(x)
    next_words[i,x] = 1

print("Data preprocessing done")
model.fit([images, partial_captions], next_words, batch_size=1, nb_epoch=5)