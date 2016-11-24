from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

from keras.optimizers import SGD

from pandas import HDFStore, DataFrame



store = HDFStore('dataset_h5/labels.h5','r')


ava_table = store['labels_train']


X_train = to_categorical(ava_table.ix[:,10:12].as_matrix())[:,1:]
Y_train = ava_table.ix[:, "good"].as_matrix()
Y_train = to_categorical(Y_train,2)

# # Group values into buckets
# Y_train = to_categorical(list(map(int,Y_train)),10)
# Y_train = to_categorical(Y_train, 2)

ava_test = store['labels_test']


X_test = to_categorical(ava_test.ix[:,10:12].as_matrix())[:,1:]
Y_test = ava_test.ix[:, "good"].as_matrix()
Y_test = to_categorical(Y_test, 2)


model = Sequential([
    Dense(32, input_dim=65),
    Activation('relu'),
    Dense(64),
    Activation('relu'),
    Dense(64),
    Activation('relu'),
    Dense(2),
    Activation('softmax'),
])

sgd = SGD(lr=0.01, decay=5e-4, momentum=0.9, nesterov=True)
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, Y_train, nb_epoch=10, batch_size=32)

model.evaluate(X_test,Y_test)