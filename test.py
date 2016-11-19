import numpy
import h5py

from fuel.datasets import H5PYDataset
sizes = numpy.random.randint(3, 9, size=(100,))
train_image_features = [
     numpy.random.randint(256, size=(3, size, size)).astype('uint8')
     for size in sizes[:90]]
test_image_features = [
     numpy.random.randint(256, size=(3, size, size)).astype('uint8')
     for size in sizes[90:]]


f = h5py.File('dataset.hdf5', mode='w')
# f['vector_features'] = numpy.vstack(
#     [numpy.load('train_vector_features.npy'),
#      numpy.load('test_vector_features.npy')])
# f['targets'] = numpy.vstack(
#     [numpy.load('train_targets.npy'),
#      numpy.load('test_targets.npy')])
# f['vector_features'].dims[0].label = 'batch'
# f['vector_features'].dims[1].label = 'feature'
# f['targets'].dims[0].label = 'batch'
# f['targets'].dims[1].label = 'index'



all_image_features = train_image_features + test_image_features
dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
image_features = f.create_dataset('image_features', (100,), dtype=dtype)
image_features[...] = [image.flatten() for image in all_image_features]
image_features.dims[0].label = 'batch'


image_features_shapes = f.create_dataset(
    'image_features_shapes', (100, 3), dtype='int32')
image_features_shapes[...] = numpy.array(
    [image.shape for image in all_image_features])
image_features.dims.create_scale(image_features_shapes, 'shapes')
image_features.dims[0].attach_scale(image_features_shapes)




image_features_shape_labels = f.create_dataset(
    'image_features_shape_labels', (3,), dtype='S7')
image_features_shape_labels[...] = [
    'channel'.encode('utf8'), 'height'.encode('utf8'),
    'width'.encode('utf8')]
image_features.dims.create_scale(
    image_features_shape_labels, 'shape_labels')
image_features.dims[0].attach_scale(image_features_shape_labels)


split_dict = {
    'train': {'image_features': (0, 90)},
    'test': {'image_features': (90, 100)}}
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f.flush()
f.close()


train_set = H5PYDataset(
    'dataset.hdf5', which_sets=('train',), sources=('image_features',))
print(train_set.axis_labels['image_features'])


handle = train_set.open()
images, = train_set.get_data(handle, slice(0, 10))
train_set.close(handle)
print(images[0].shape, images[1].shape)