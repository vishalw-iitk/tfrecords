"""
  Input pipeline unittest
"""

import unittest
import tensorflow as tf

import trainer.input_pipeline as input_pipeline

IMAGE_SIZE = 28
NUM_CLASSES = 10
BUCKET_NAME = 'qodelabs'
BATCH_SIZE = 8
NUM_EPOCHS = 2          # Dummy value
TFRECORD_DIR = 'fashion_mnist/tfrecord/'
NUM_WORKERS = 6         # For shuffle buffer size

SESS = tf.InteractiveSession()

class DataTest(unittest.TestCase):
  """
    Unittests for test output of input pipeline
  """
  def setUp(self):
    """
      Data test class initialization
    """
    self.data = input_pipeline.TfrecordInput(
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        bucket_name=BUCKET_NAME,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        data_dir=TFRECORD_DIR,
        num_workers=NUM_WORKERS)

  @staticmethod
  def get_data_batch(data_input_fn):
    """
      Get batch of data
    """
    dataset = data_input_fn()
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch = iterator.get_next()
    return batch

  def test_data(self):
    """
      Test output data as required from pipeline
    """
    data_input_fn = self.data.data_input_fn()
    batch = self.get_data_batch(data_input_fn)
    features, labels = batch
    features = features['input_image'].eval(session=SESS)
    labels = labels.eval(session=SESS)
    features_required_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    labels_required_shape = (BATCH_SIZE, 1, NUM_CLASSES)
    self.assertTupleEqual(features.shape, features_required_shape)
    self.assertTupleEqual(labels.shape, labels_required_shape)

if __name__ == "__main__":
  unittest.main()
