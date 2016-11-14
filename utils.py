from shutil import copyfile
import pandas as pd
from pandas import HDFStore
import os

def copyTestSet(dest_dir="test_images/"):
  ava_path = "dataset/AVA/data/"
  ava_data_path = os.path.join(os.getcwd(), ava_path)
  store = HDFStore('dataset_h5/labels.h5')
  test_labels = store['labels_test']
  imagesCount = test_labels.shape[0]

  i = 0
  for index, row in test_labels.iterrows():
    if (i % 1000) == 0:
      print("Now processing {} / {}".format(i,imagesCount))
    filename = str(index) + ".jpg"
    filepath = os.path.join(ava_data_path, filename)
    copyfile(filepath, dest_dir + filename)
    i=i+1