import os
import tensorflow as tf

tfrecord_location = '/home/vishal/Desktop/mnist read tfrefcord/tfr'
name = "part-00000-of-00005.tfrecord"
filename = os.path.join(tfrecord_location, name)

dataset = tf.data.TFRecordDataset(filename)

def decode(serialized_example):
  """
  Parses an image and label from the given `serialized_example`.
  It is used as a map function for `dataset.map`
  """
  IMAGE_SIZE = 28
  IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
  
  # 1. define a parser
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          '': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.string),
          'height': tf.FixedLenFeature([], tf.string),
          'width': tf.FixedLenFeature([], tf.string),
          'depth': tf.FixedLenFeature([], tf.string),

      })

  # 2. Convert the data
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  label = tf.cast(features['label'], tf.int32)
  # 3. reshape
  image.set_shape((IMAGE_PIXELS))
  return image, label

dataset = dataset.map(decode)

def normalize(image, label):
  """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  return image, label

dataset = dataset.map(normalize)
batch_size = 1000
dataset = dataset.shuffle(1000 + 3 * batch_size )
dataset = dataset.repeat(2)
dataset = dataset.batch(batch_size)


iterator = dataset.make_one_shot_iterator()
image_batch, label_batch = iterator.get_next()


sess = tf.Session()
image_batch, label_batch = sess.run([image_batch, label_batch])
print(image_batch.shape)
print(label_batch.shape)



