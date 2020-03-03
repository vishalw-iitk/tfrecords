"""
  Estimator run Configuration
"""

import tensorflow as tf
from tensorflow.estimator import RunConfig

# PS_STRATEGY = tf.distribute.experimental.ParameterServerStrategy()
# MWM_STRATEGY = tf.distribute.experimental.MultiWorkerMirroredStrategy()
# M_STRATEGY = tf.contrib.distribute.MirroredStrategy()

SESS_CONFIG = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    intra_op_parallelism_threads=0,
    gpu_options=tf.GPUOptions(force_gpu_compatible=True))

def get_strategy(strategy):
  '''
  Returns distribution strategy according to required time
  ARGS:
    strategy="mirror" or "parameter_server"
  '''
  if strategy == "parameter_server":
    strategy = tf.distribute.experimental.ParameterServerStrategy()
  elif strategy == "mirror":
    strategy = tf.contrib.distribute.MirroredStrategy()
  else:
    strategy = None
  return strategy


def get_run_config(strategy):
  """
    Get Estimator run config
    Returns:
      Type: RunConfig
  """
  strategy = get_strategy(strategy)

  config = RunConfig(model_dir=None,
                     tf_random_seed=None,
                     save_summary_steps=10,
                     save_checkpoints_steps=20,
                     session_config=SESS_CONFIG,
                     keep_checkpoint_max=5,
                     log_step_count_steps=100,
                     train_distribute=strategy,
                     device_fn=None,
                     protocol=None,
                     eval_distribute=strategy,
                     experimental_distribute=None,
                     experimental_max_worker_delay_secs=None)
  return config
