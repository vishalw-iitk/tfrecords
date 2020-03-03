"""
  Serving input receiver function:
  For online and batch prediction
"""

import tensorflow as tf

def get_online_serving_input_receiver_fn(image_size):
  """
    Decorator for Serving input function for online predictions
  """
  def _online_serving_input_receiver_fn():
    """
      Serving input function for online predictions
    """
    receiver_tensors = {
        'input_image': tf.placeholder(dtype=tf.float32,
                                      shape=[None, image_size,
                                             image_size, 3],
                                      name='input_image')
    }
    features = {
        'input_image': receiver_tensors['input_image']
    }
    return tf.estimator.export.ServingInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors)
  return _online_serving_input_receiver_fn

def get_online_final_exporter(image_size):
  """
    Get final exporter for online predictions
  """
  online_final_exporter = tf.estimator.FinalExporter(
      'online_final_exporter',
      get_online_serving_input_receiver_fn(image_size))
  return online_final_exporter


def get_cloud_serving_input_receiver_fn(image_size):
  """
    Decorator for batch predictions serving function
  """
  def _cloud_serving_input_receiver_fn():
    """
      Serving function for batch predictions
    """
    receiver_tensors = tf.placeholder(dtype=tf.string)
    feature = {'test' + '/image': tf.FixedLenFeature([], tf.string),
               'test' + '/label': tf.FixedLenFeature([], tf.string)}
    features = tf.parse_example(receiver_tensors, features=feature)
    image = tf.decode_raw(features['test' + '/image'], tf.float32)
    image = tf.reshape(
        image, [-1, image_size, image_size, 3]) # HARDCODE need to rectify
    image = {'input_image': image}
    return tf.estimator.export.ServingInputReceiver(
        features=image,
        receiver_tensors=receiver_tensors)
  return _cloud_serving_input_receiver_fn

def get_cloud_final_exporter(image_size):
  """
    Get final exporter for online predictions
  """
  cloud_final_exporter = tf.estimator.FinalExporter(
      'cloud_final_exporter',
      get_cloud_serving_input_receiver_fn(image_size))
  return cloud_final_exporter
