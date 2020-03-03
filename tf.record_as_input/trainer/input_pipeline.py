"""
    Tfrecord input pipeline
"""
# pylint: disable=R0902, R0913, E1101,

import os

import tensorflow as tf
from google.cloud import storage
import utils

from serving_fn import get_online_final_exporter,\
    get_cloud_final_exporter

CLIENT = storage.Client()


class TfrecordInput():
  """
      Class for tfrecord input-pipeline
  """

  def __init__(self,
               data_dir,
               bucket_name,
               num_workers,
               image_size=None,
               num_classes=None,
               batch_size=64, num_epochs=1,
               mode='train'):
    """
        Initialization
    """
    self.data_dir = utils.make_valid_dir(data_dir)
    self.bucket_name = bucket_name
    self.image_size = image_size
    self.num_classes = num_classes
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.mode = mode
    self.list_of_data_path = self.make_list_of_data_path(
        self.data_dir + utils.make_valid_dir(self.mode))
    self.num_workers = num_workers
    # self.iterator = self.create_iterator()

  def make_list_of_data_path(self, data_dir):
    """
        Make list of tfrecords path from given data_dir
    """
    bucket = CLIENT.get_bucket(self.bucket_name)
    list_of_data_path = [os.path.join("gs://"+self.bucket_name+\
                         '/', f.name) for f in
                         bucket.list_blobs(prefix=data_dir)]
    return list_of_data_path

  def parser(self):
    """
        Parser for tf records
    """
    def inner_parser(record):
      """
          Inner parser function to decode tfrecord sample
      """
      feature = {self.mode + '/image': tf.FixedLenFeature([], tf.string),
                 self.mode + '/label': tf.FixedLenFeature([], tf.string)}
      features = tf.parse_single_example(record, features=feature)
      label = tf.decode_raw(features[self.mode + '/label'], tf.float32)
      label = tf.reshape(label, [1, self.num_classes])
      image = tf.decode_raw(features[self.mode + '/image'], tf.float32)
      image = tf.reshape(
          image, [self.image_size, self.image_size, 3])
      image = {'input_image': image}
      print("image", image)
      print("label", label)
      return image, label
    return inner_parser

  def create_iterator(self):
    """
        Make one-shot iterator for data
    """
    dataset = self.create_tf_record_data()
    iterator = dataset.make_one_shot_iterator()
    return iterator

  def create_tf_record_data(self):
    """
        Create tensorflow dataset from tf records
    """
    dataset = tf.data.TFRecordDataset(self.list_of_data_path)
    dataset = dataset.map(self.parser())
    dataset = dataset.shuffle(
        buffer_size=self.num_workers*self.batch_size)
    dataset = dataset.batch(self.batch_size, drop_remainder=True)
    dataset = dataset.repeat(self.num_epochs)
    return dataset

  def data_input_fn(self):
    """
      Input function which returns dataset
    """
    def _data_input_fn():
      """
        Inner data input function
      """
      return self.create_tf_record_data()
    return _data_input_fn

  def input_fn(self):
    """
        Input function for estimator
    """
    def inner_input_fn():
      """
          Inner input function
      """
      iterator = self.create_iterator()
      features, labels = iterator.get_next()
      labels = tf.reshape(labels, (self.batch_size, self.num_classes))
      return features, labels
    return inner_input_fn

  def get_exporters(self):
    """
      Get exporters
    """
    online_final_exporter = get_online_final_exporter(self.image_size)
    cloud_final_exporter = get_cloud_final_exporter(self.image_size)
    return [online_final_exporter, cloud_final_exporter]
