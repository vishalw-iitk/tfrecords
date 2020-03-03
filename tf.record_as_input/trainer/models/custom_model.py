"""
  Custom model
"""

# pylint: disable = E0401

import tensorflow as tf

# Custom imports
from tensorflow.keras.layers import Conv2D, MaxPool2D

class ModelArchitecture():
  """
    Custom model Architecture
  """
  def __init__(self, mode):
    """
      Initialization
    """
    self.mode = mode
    assert (not mode == 'transfer'),\
      "transfer functionality is not supported in Custom model"

  @staticmethod
  def _custom_net(features):
    """
      Custom CNN model
    """
    # Write your custom model here
    layer = Conv2D(32, (3, 3),
                   input_shape=(None, 224, 224, 3),
                   padding='valid',
                   activation='relu')(features)
    layer = MaxPool2D()(layer)
    layer = Conv2D(64, (3, 3),
                   padding='valid',
                   activation='relu')(layer)
    layer = MaxPool2D()(layer)
    return layer

  @staticmethod
  def prepare_input(features):
    """
        Preprocess input features as required
        for custom model input layer
    """
    # features = tf.keras.utils.normalize(features)
                 # Bug: Tensor object has no attribute conjugate,
                 # But works on eager execution

    features = tf.image.resize(features, size=(224, 224))
    return features

  def compute_net(self, features):
    """
        Compute Custom Net depending on mode
    """
    return self._custom_net(features)
