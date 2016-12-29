import os
import h5py
import cv2
import numpy as np

periodNum = len(os.listdir('images/'))

with h5py.File('selfie_images_224x224.h5','w') as f:
    data = f.create_dataset("data",(periodNum,3,224,224), dtype='uint8')
    for i, filename in enumerate(os.listdir('images/')):
        if filename.endswith(".jpg"): 
            im = cv2.resize(cv2.imread('images/{}'.format(filename)), (224, 224)).astype(np.float32)
            im[:,:,0] -= 103.939
            im[:,:,1] -= 116.779
            im[:,:,2] -= 123.68
            im = im.transpose((2,0,1))
            im = np.expand_dims(im, axis=0)
            data[i] = im
        i = i + 1
