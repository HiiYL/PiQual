from keras.models import Sequential
from scipy import ndimage, misc
import numpy as np

model = Sequential()

from keras.layers import Dense, Activation

model.add(Dense(output_dim=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

X_train = 
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)


images = np.array
ava_dataset = "dataset/AVA/data"
root_path = os.getcwd()
ava_data_path = os.path.join(root_path, ava_dataset)

count = 10000#len(os.listdir(ava_data_path))
images = np.empty(count, dtype=object)
i=0
print "Loading Images..."
for root, dirnames, filenames in os.walk(ava_data_path):
    for filename in filenames:
      if i >= count:
        break
      if (i % 1000) == 0:
        print "Now processing " + str(i) + "/" + str(count)

      filepath = os.path.join(root, filename)
      image = ndimage.imread(filepath, mode="RGB")
      image_resized = misc.imresize(image, (64, 64))
      images[i] = image_resized
      i=i+1