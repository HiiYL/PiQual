from pandas import HDFStore
import pandas as pd
import numpy as np
import os

def good(c):
  if c['score'] >= 5:
    return 1
  else:
    return 0


## Define dataset paths
ava_path = "dataset/AVA/data/"
ava_data_path = os.path.join(os.getcwd(), ava_path)

## Read AVA table
ava_table = pd.read_table("dataset/AVA/AVA.txt", delim_whitespace=True, index_col=0,
header=None,usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

## Check for missing files
labels_count = ava_table.shape[0]
count=0
invalid_indices = []
for index, row in ava_table.iterrows():
    if (count % 1000) == 0:
        print('Now checking {0}/{1}'.format(count,labels_count))
    filename = str(index) + ".jpg"
    filepath = os.path.join(ava_data_path, filename)
    if not os.path.isfile(filepath):
        invalid_indices.append(index)
        print('{0} at position {1} is missing or invalid'.format(filename, count))
    count = count + 1

if invalid_indices:
    ava_table = ava_table.drop(invalid_indices)
    store['labels'] = ava_table


## Calculate mean rating
sum_of_ratings = (ava_table.ix[:,:10]).sum(axis=1)
weights = np.array([1,2,3,4,5,6,7,8,9,10])
score = (ava_table.ix[:,:10] * weights).sum(axis=1) / sum_of_ratings

ava_table['score'] = score


## Calculate standard deviation for probability prediction
standard_deviation = np.sqrt(((ava_table.ix[:,:10] * np.square(weights)).sum(axis=1) / sum_of_ratings) - (ava_table.score ** 2))
ava_table['standard_deviation'] = standard_deviation

## Binarize score to good/bad
ava_table['good'] = ava_table.apply(good, axis=1)

## Save modified AVA dataset to HDFStore
store = HDFStore('dataset/labels.h5')
store['labels'] = ava_table

## Preprocess Comments
ava_comment_path = "dataset/AVA-Comments/"
ava_comment_path = os.path.join(os.getcwd(), ava_comment_path)
comments_series = pd.Series(index=ava_table.index)

store = HDFStore('dataset_h5/labels.h5')


## Load comments and save
labels_count = ava_table.shape[0]
count=0
for index, row in ava_table.iterrows():
    if (count % 1000) == 0:
        print('Now Processing Comments {0}/{1}'.format(count,labels_count))
    filename = str(index) + ".txt"
    filepath = os.path.join(ava_comment_path, filename)
    with open(filepath,encoding = "ISO-8859-1") as f:
        content = f.readlines()
        stripped_contents = [ string.strip() for string in content ]
        comments_series.ix[index] = stripped_contents
    count = count + 1
ava_table.loc[:,'comments'] = comments_series
