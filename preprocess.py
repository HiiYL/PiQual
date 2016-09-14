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

sum_of_ratings = (ava_table.ix[:,:10]).sum(axis=1)
weights = np.array([1,2,3,4,5,6,7,8,9,10])
score = (ava_table.ix[:,2:11] * weights).sum(axis=1) / sum_of_ratings

ava_table['score'] = score

standard_deviation = np.sqrt(((ava_table.ix[:,:10] * np.square(weights)).sum(axis=1) / sum_of_ratings) - (ava_table.score ** 2))
ava_table['standard_deviation'] = standard_deviation

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



###########preprocess comments
ava_comment_path = "dataset/AVA-Comments/"
ava_data_path = os.path.join(os.getcwd(), ava_comment_path)
comments_series = pd.Series(index=ava_table.index)

store = HDFStore('dataset_h5/labels.h5')

labels_count = ava_table.shape[0]
count=0
for index, row in ava_table.iterrows():
    if (count % 1000) == 0:
        print('Now Processing Comments {0}/{1}'.format(count,labels_count))
    filename = str(index) + ".txt"
    filepath = os.path.join(ava_data_path, filename)
    with open(filepath,encoding = "ISO-8859-1") as f:
        content = f.readlines()
        stripped_contents = [ string.strip() for string in content ]
        comments_series.ix[index] = stripped_contents
    count = count + 1
ava_table.loc[:,'comments'] = comments_series


import sys
import numpy as np
import gensim
from word2veckeras.word2veckeras import Word2VecKeras

def compare_w2v(w2v1,w2v2):
    s=0.0
    count =0
    for w in w2v1.vocab:
        if w in w2v2.vocab:
            d=np.linalg.norm(w2v1[w]-w2v2[w])
            count +=1
            s += d
    return s/count


input_file = 'test.txt'
sents=gensim.models.word2vec.LineSentence(input_file)

v_iter=1
v_size=5
sg_v=1
topn=4

vs1 = gensim.models.word2vec.Word2Vec(sents,hs=1,negative=0,sg=sg_v,size=v_size,iter=1)
                      
print vs1['the']
vsk1 = Word2VecKeras(sents,hs=1,negative=0,sg=sg_v,size=v_size,iter=1)
print( vsk1.most_similar('the', topn=topn))
print vsk1['the']
print np.linalg.norm(vs1.syn0-vsk1.syn0),compare_w2v(vs1,vsk1)
vsk1 = Word2VecKeras(sents,hs=1,negative=0,sg=sg_v,size=v_size,iter=5)
print vsk1['the']
print( vsk1.most_similar('the', topn=topn))
print( vs1.most_similar('the', topn=topn))
print np.linalg.norm(vs1.syn0-vsk1.syn0),compare_w2v(vs1,vsk1)
