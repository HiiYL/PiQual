from pandas import HDFStore
import pandas as pd

def good(c):
  if c['score'] >= 5:
    return 1
  else:
    return 0

ava_table = pd.read_table("AVA/AVA.txt", delim_whitespace=True, index_col=0,
header=None,usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

sum_of_ratings = (ava_table.ix[:,2:11]).sum(axis=1)
weights = [1,2,3,4,5,6,7,8,9,10]
score = (ava_table.ix[:,2:11] * weights).sum(axis=1) / sum_of_ratings

ava_table['score'] = score

ava_table['good'] = ava_table.apply(good, axis=1)

store = HDFStore('labels.h5')

store['labels'] = ava_table


## Check for missing files
for index, row in ava_table.iterrows():
    if (i % 1000) == 0:
        print "Now processing " + str(i) + "/" + str(labels_count)
    filename = str(index) + ".jpg"
    filepath = os.path.join(ava_data_path, filename)
    if not os.path.isfile(filepath):
        invalid_indices.append(index)
        print filename + " at position " + str(i) + " is missing or invalid."
    i = i + 1
if invalid_indices:
    ava_table = ava_table.drop(invalid_indices)
    store['labels'] = ava_table