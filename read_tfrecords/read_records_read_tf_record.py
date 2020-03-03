# pylint: disable=W0612, R0913
'''File that reads Tfrecord and display images and labels'''
import os
import argparse
import numpy as np
from google.cloud import storage
import tensorflow as tf
import matplotlib.pyplot as plt

CLIENT = storage.Client()


def get_args():
  """Define the task arguments with the default values.
  Returns:
    experiment parameters
  """
  args_parser = argparse.ArgumentParser()
  args_parser.add_argument(
      '--num-classes',
      help="Number of classes",
      default = 100,
      #required=True,
      type=int
  )
  # args_parser.add_argument(
  #     '--image-size',
  #     help='Height and width of image',
  #     required=True,
  #     type=int
  # )
  args_parser.add_argument(
      '--batch-size',
      help='Batch size for each training and evaluation step.',
      type=int,
      default=2,
  )
  args_parser.add_argument(
      '--mode',
      help="""
            train, test, valid
        """,
      default = 'train',
      #required=True,
      type=str,
      #choices=['train', 'test', 'valid']
  )
  args_parser.add_argument(
      '--num-batch',
      help='num of batches',
      type=int,
      default=3
  )
  args_parser.add_argument(
      '--data-path',
      help="""
            path for tfrecord
        """,
      default = '/home/vishal/Desktop/tf_data_path',
      #required=True,
      type=str
  )
  args_parser.add_argument(
      '--bucket-name',
      help="""
            GCS bucket name
        """,
      default = 'qommunicator',
      #required=True,
      type=str
  )
  return args_parser.parse_args()


class ReadTfrecord():
  '''
  Contains functions used in tfrecord
  '''

  def __init__(self, data_path,
               num_classes,
               #img_size,
               mode,
               batch_size,
               num_batch,
               bucket_name):
    '''
    Constructor for class read tfrecord
    '''
    self.datapath = data_path
    self.num_classes = num_classes
    #self.img_size = img_size
    self.mode = mode
    self.batch_size = batch_size
    self.num_batches = num_batch
    self.bucket_name = bucket_name

  def get_images_and_labels(self):
    '''
    Reads the tfrecord and returns images and labels
    '''
    # feature = {self.mode + '/image': tf.FixedLenFeature([], tf.string),
    #            self.mode + '/label': tf.FixedLenFeature([], tf.string)}
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(self.datapath[0])

    for image_count in range(10):
    	path = str(image_count)
    	feature_dict = {path: tf.FixedLenFeature([], tf.string),
    	'height': tf.FixedLenFeature([], tf.int64),
    	'width': tf.FixedLenFeature([], tf.int64),
    	'depth': tf.FixedLenFeature([], tf.int64)}

    	features = tf.parse_single_example(serialized_example, features=feature_dict)
    	image_buffer = tf.reshape(features[path], shape=[])
    	image = tf.decode_raw(image_buffer, tf.uint8)
    	image = tf.reshape(image, tf.stack([height, width, num_depth]))
    	image = tf.reshape(image, [1, height, width, num_depth])
    	image_seq.append(image)

    # image = tf.reshape(image, [60,60,4])
    

    label = tf.reshape(label, [1, self.num_classes])
    return image_seq, label

  def show_images(self, tf_session, images, labels):
    '''
    Function to display images
    ARGS :
        sess : tensorflow session
        images : Images
        labels : one hot encoded
    '''
    for batch_index in range(self.num_batches):
      img, lbl = tf_session.run([images, labels])
      lbl = np.argmax(lbl, axis=0)
      print(lbl)
      img = img.astype(np.uint8)
      for j in range(self.batch_size):
        plt.subplot(2, 3, j + 1)
        plt.imshow(img[j, ...])
        # plt.title(lbl[j])
      plt.show()

  def make_list_of_data_path(self):
    """
        Make list of tfrecords path from given data_dir
    """
    #bucket = CLIENT.get_bucket(self.bucket_name)
    list_of_data_path = [os.path.join(self.datapath,f) for f in os.listdir(self.datapath)]
    # list_of_data_path = [os.path.join("gs://" + self.bucket_name +
    #                                   '/', f.name) for f in
    #                      bucket.list_blobs(prefix=self.datapath)]
    return list_of_data_path


def main(tf_session):
  '''
  Main function to read and show tfrecord
  '''
  args = get_args()
  read = ReadTfrecord(data_path=args.data_path,
                      num_classes=args.num_classes,
                      #img_size=args.image_size,
                      mode=args.mode,
                      batch_size=args.batch_size,
                      num_batch=args.num_batch,
                      bucket_name=args.bucket_name,
                      )
  image, label = read.get_images_and_labels()
  print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",image,"***",label)
  images, labels = tf.train.shuffle_batch([image, label],
                                          batch_size=args.batch_size,
                                          capacity=20,
                                          allow_smaller_final_batch=True,
                                          #num_threads=1)
                                          min_after_dequeue=1)
  
  init_op = tf.group(
      tf.global_variables_initializer(),
      tf.local_variables_initializer())
  tf_session.run(init_op)
  
  coord = tf.train.Coordinator()
  tf.train.start_queue_runners(coord=coord)
  print("$$$@@@@@@@@@@@",sess,"__",images,"__",label)
  read.show_images(sess, images, labels)


if __name__ == "__main__":
  with tf.Session() as sess:
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    main(sess)

