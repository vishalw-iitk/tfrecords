"""
    Model Functions
"""
# pylint: disable = E0401, R1705, R1710, R0902, R0913, E1101, E0611

# import os

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten  # , Conv2D

from models.mobile_net import ModelArchitecture as MobileNet
from models.resnet import ModelArchitecture as ResNet
from models.vgg16 import ModelArchitecture as VGG
from models.xception import ModelArchitecture as Xception
from models.custom_model import ModelArchitecture as Custom

from run_config import get_run_config

import utils
import metrics

tf.logging.set_verbosity(tf.logging.INFO)

MODELS = {'ResNet': ResNet,
          'VGG': VGG,
          'MobileNet': MobileNet,
          'Xception': Xception,
          'Custom': Custom}


def get_logging_hooks(loss):
  """
      Get logging hook for training estimator
  """
  logging_hook = tf.train.LoggingTensorHook({'loss': loss},
                                            every_n_iter=10)
  return logging_hook


class ClassificationEstimator():
  """
      Class with MobileNet functionalities
      Functionalities: Train from scratch, Retrain,
                       Fine-tune over pretrained weights
  """

  def __init__(self,
               num_class,
               batch_size,
               model_name,
               input_fn,
               learning_rate,
               train_steps,
               eval_steps,
               strategy,
               eval_input_fn=None,
               exporters=None,
               eval_throttle_secs=600
               ):
    """
        Initialization
        Args:
            mode = train, retrain, transfer
    """
    self.mode = None
    self.num_class = num_class
    self.batch_size = batch_size
    self.input_fn = input_fn
    self.eval_input_fn = eval_input_fn
    self.model_name = model_name
    self.learning_rate = learning_rate
    self.train_steps = train_steps
    self.eval_steps = eval_steps
    self.exporters = exporters
    self.strategy = strategy
    self.eval_throttle_secs = eval_throttle_secs

  def compute_graph(self, features):
    """
        Compute mobilenet graph
    """
    layer = MODELS[self.model_name](mode=self.mode).compute_net(features)

    layer = Flatten()(layer)

    # layer = Dense(1024, activation='relu')(layer)
    # layer = Dense(512, activation='relu')(layer)
    preds = Dense(self.num_class, activation='softmax')(layer)
    print("compute-------------------graph")
    print(preds)
    return preds

  def model_fn(self):
    """
        Model function decorator
    """
    def model_function(features, labels, mode):
      """
          Estimator model function
      """
      features = MODELS[self.model_name].prepare_input(features['input_image'])
      preds = self.compute_graph(features)
      predictions = {
          'class': tf.argmax(input=preds, axis=1),
          'probabilities': preds
      }
      export_outputs = {
          'prediction': tf.estimator.export.PredictOutput(predictions)
      }
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions,
            export_outputs=export_outputs)

      labels = tf.reshape(labels, (self.batch_size, self.num_class))
      # Calculate Loss and Accuracy (for both TRAIN and EVAL modes)
      loss = tf.math.reduce_mean(
          tf.keras.losses.categorical_crossentropy(
              y_true=labels, y_pred=preds))

      accuracy = metrics.custom_accuracy(labels, preds)
      tf.summary.scalar('custom_accuracy', tf.math.reduce_mean(accuracy))

      # Configure the Training Op (for TRAIN mode)
      if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        grads = optimizer.compute_gradients(loss)
        # tf.summary.histogram('gradients', grads)
        for grad in grads:
          tf.summary.histogram("%s-grad" % grad[1].name, grad[0])
        tf.summary.merge_all()

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op,
            training_hooks=[get_logging_hooks(loss)])

      # Add evaluation metrics (for EVAL mode)
      eval_metric_ops = {
          'accuracy': accuracy}
      return tf.estimator.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        eval_metric_ops=eval_metric_ops)
    return model_function

  def train_estimator(self, model_dir):
    """
        Create and train estimator
    """
    custom_estimator_model = tf.estimator.Estimator(
        model_fn=self.model_fn(), model_dir=model_dir,
        config=get_run_config(self.strategy))
    train_spec = tf.estimator.TrainSpec(input_fn=self.input_fn,
                                        max_steps=self.train_steps)
    assert (self.eval_input_fn), "Please provide eval input function"
    eval_spec = tf.estimator.EvalSpec(input_fn=self.eval_input_fn,
                                      steps=self.eval_steps,
                                      exporters=self.exporters,
                                      throttle_secs=self.eval_throttle_secs)
    tf.estimator.train_and_evaluate(custom_estimator_model,
                                    train_spec,
                                    eval_spec)

  def train_from_scratch(self, model_dir):
    """
        Train model from scratch
    """
    self.mode = 'train_from_scratch'
    # assert (not tf.gfile.Exists(model_dir)),\
    #     "Model directory: '" + str(model_dir) + "' already exists,\n" +\
    #     "Provide new model directory for training model from scratch"
    self.train_estimator(model_dir)

  def retrain(self, model_dir):
    """
        Retrain model graph
    """
    model_dir = utils.make_valid_dir(model_dir)
    self.mode = 'retrain'
    assert (tf.gfile.Exists(model_dir)),\
        "Model directory: '" + str(model_dir) + "' doesn't exists,\n" +\
        "Train model from scratch before retraining"
    self.train_estimator(model_dir)

  def transfer(self, model_dir):
    """
        Tune our own classifier network over
        pretrained feature extractor
    """
    model_dir = utils.make_valid_dir(model_dir)
    self.mode = 'transfer'
    self.train_estimator(model_dir)
