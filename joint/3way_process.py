store = HDFStore('../dataset_h5/labels.h5','r')

ava_table = store['labels']

ava_path = "../dataset/AVA/data/"
style = pd.read_table('../dataset/AVA/style_image_lists/train.jpgl', index_col=0)
tag = pd.read_table('../dataset/AVA/style_image_lists/train.lab')

style.loc[:,'style'] = tag.as_matrix()

ava_with_style = style.join(ava_table, how='inner')

store['labels_with_style'] = ava_with_style

data = self.h5f.create_dataset("data_style_train",(periodNum,channel,width,height), dtype='uint8')

periodNum = ava_with_style.shape[0]

for index in ava_with_style.index:
  if(i >= periodNum):
    break
  if (i % 1000) == 0:
      print('Now Processing {0}/{1}'.format(i,periodNum))
      filename = str(index) + ".jpg"
      filepath = os.path.join(self.ava_data_path, filename)
      im = cv2.resize(cv2.imread(filepath), (width, height)).astype(np.float32)
      im[:,:,0] -= 103.939
      im[:,:,1] -= 116.779
      im[:,:,2] -= 123.68
      im = im.transpose((2,0,1))
      im = np.expand_dims(im, axis=0)
      data[i] = im
      i=i+1
	
