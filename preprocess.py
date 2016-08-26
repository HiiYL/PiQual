from pandas import HDFStore
import pandas as pd
import os

def good(c):
  if c['score'] >= 5:
    return 1
  else:
    return 0

ava_table = pd.read_table("dataset/AVA/AVA.txt", delim_whitespace=True, index_col=0,
header=None,usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

sum_of_ratings = (ava_table.ix[:,2:11]).sum(axis=1)
weights = [1,2,3,4,5,6,7,8,9,10]
score = (ava_table.ix[:,2:11] * weights).sum(axis=1) / sum_of_ratings

ava_table['score'] = score

ava_table['good'] = ava_table.apply(good, axis=1)

store = HDFStore('dataset/labels.h5')

store['labels'] = ava_table


ava_path = "dataset/AVA/data/"
ava_data_path = os.path.join(os.getcwd(), ava_path)
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