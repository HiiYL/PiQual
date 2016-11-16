from os import listdir
from os.path import isfile, join
import numpy
import cv2

mypath='frames/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty((len(onlyfiles),3,224,224), dtype=np.float32)

for n in range(0, len(onlyfiles)):
  im = cv2.resize(cv2.imread( join(mypath,onlyfiles[n]) ), (224, 224)).astype(np.float32)
  im[:,:,0] -= 103.939
  im[:,:,1] -= 116.779
  im[:,:,2] -= 123.68
  im = im.transpose((2,0,1))
  im = np.expand_dims(im, axis=0)
  images[n] = im


prediction = model.predict(images)

top_frames = np.argsort(prediction[:,1])


cv2.imshow('image',cv2.imread('frames/' +onlyfiles[508],0))
cv2.waitKey(0)