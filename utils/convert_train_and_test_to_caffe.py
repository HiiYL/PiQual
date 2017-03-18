from pandas import HDFStore


store = HDFStore('dataset/labels.h5')
ava_table = store['labels_train']

path = [ '/home/hii/Projects/PiQual/dataset/AVA/data/' + str(value) + '.jpg' for value in ava_table.index ]

ava_table.loc[:,'path'] = path

ava_table[['path','good']].to_csv('train.txt',header=None, index=None,sep=' ')

ava_test = store['labels_test']

path = [ '/home/hii/Projects/PiQual/dataset/AVA/data/' + str(value) + '.jpg' for value in ava_test.index ]

ava_test.loc[:,'path'] = path

ava_test[['path','good']].to_csv('test.txt',header=None, index=None,sep=' ')