"""
  Custom estimator metrics
"""

import tensorflow as tf


def random_metric(labels, predictions):
  """
    Random estimator metrics
  """
  sub = tf.subtract(labels, predictions)
  return tf.metrics.mean(sub)

def recall(y_true, y_pred):
  """
    Custom metric for mulit-label recall
    TN / (TN+FN)
  """
  threshold = tf.convert_to_tensor(0.3)
  y_p = tf.dtypes.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)
  num_ones_pred = tf.math.count_nonzero(y_p)
  total = tf.dtypes.cast(tf.size(y_p), num_ones_pred.dtype)
  num_zeros_pred = tf.subtract(total, num_ones_pred) # TN + FN
  true_negetives = tf.subtract(total,
                               tf.math.count_nonzero(tf.add(y_p, y_true)))
  return tf.metrics.mean(tf.divide(true_negetives, num_zeros_pred))

def custom_accuracy(y_true, y_pred):
  """
    Custom Accuracy
  """
  label_classes = tf.argmax(y_true, axis=1)
  pred_classes = tf.argmax(y_pred, axis=1)
  correct_preds = tf.math.count_nonzero(
      tf.dtypes.cast(tf.math.equal(label_classes, pred_classes), tf.int8))
  correct_preds = tf.dtypes.cast(correct_preds, tf.float32)
  total_count = tf.dtypes.cast(tf.size(pred_classes), tf.float32)
  return tf.metrics.mean(tf.divide(correct_preds, total_count))
