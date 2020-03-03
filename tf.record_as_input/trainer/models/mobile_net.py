"""
    MobileNet Model
"""
# pylint: disable = E0401

import tensorflow as tf

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input

from tensorflow.keras.layers import GlobalAveragePooling2D

class ModelArchitecture():
  """
      Architecture for MobileNet
  """

  def __init__(self, mode):
    """
        Initialization
    """
    self.mode = mode

  @staticmethod
  def prepare_input(features):
    """
        Preprocess input features as required
        for MobileNet input layer
    """
    features = preprocess_input(features)
    features = tf.image.resize(features, size=(224, 224))
    return features

  def compute_net(self, features):
    """
        Compute MobileNet depending on mode
    """
    if self.mode == 'train_from_scratch' or self.mode == 'retrain':
      model_graph = MobileNet(weights=None, include_top=False)
    if self.mode == 'transfer':
      model_graph = MobileNet(include_top=False)
      for layer in model_graph.layers:
        layer.trainable = False
    layer = model_graph(features)
    layer = GlobalAveragePooling2D()(layer)
    return layer
