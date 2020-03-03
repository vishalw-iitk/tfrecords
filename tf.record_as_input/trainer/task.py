'''
Main file to run trainer package
'''

# pylint: disable=E0401, E1101, E0611, E1123

import os
import json
import argparse
from input_pipeline import TfrecordInput
from models.models import ClassificationEstimator

from tensorflow.contrib import training

def get_args():
  """Define the task arguments with the default values.
  Returns:
      experiment parameters
  """
  args_parser = argparse.ArgumentParser()

  # Experiment arguments
  args_parser.add_argument(
      '--train-steps',
      help="""
      Steps to run the training job for.
      If --num-epochs and --train-size are not specified, this must be.
      Otherwise the training job will run indefinitely.
      if --num-epochs and --train-size are specified,
      then --train-steps will be: (train-size/train-batch-size) * num-epochs
      """,
      default=10,
      type=int
  )
  args_parser.add_argument(
      '--eval-steps',
      help="""
      Number of steps to run evaluation for at each checkpoint.',
      Set to None to evaluate on the whole evaluation data.
      """,
      default=5,
      type=int
  )
  args_parser.add_argument(
      '--num-classes',
      help="""
      Number of classes
      """,
      default= 100,
      #required=True,
      type=int
  )
  args_parser.add_argument(
      '--image-size',
      help="""
      size of the image height of image for example if image is',
      of size 124*124 the input should be --image-size 124
      """,
      default= 60,
      #required=True,
      type=int
  )
  args_parser.add_argument(
      '--batch-size',
      help='Batch size for each training and evaluation step.',
      type=int,
      default=10
  )
  
  args_parser.add_argument(
      '--num-epochs',
      help="""\
        Maximum number of training data epochs on which to train.
        If both --train-size and --num-epochs are specified,
        --train-steps will be: (train-size/train-batch-size) * num-epochs.\
        """,
      default=100,
      type=int,
  )
  args_parser.add_argument(
      '--eval-throttle-secs',
      help='How many seconds to wait before running the next evaluation.',
      default=600,
      type=int
  )

  # Estimator arguments
  args_parser.add_argument(
      '--learning-rate',
      help="Learning rate value for the optimizers.",
      default=0.001,
      type=float
  )
  args_parser.add_argument(
      '--learning-rate-decay-factor',
      help="""
      The factor by which the learning rate should decay by the end of the training.
      decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps).
      If set to 1.0 (default), then no decay will occur.
      If set to 0.5, then the learning rate should reach 0.5 of its original value at the end of the training.
      Note that decay_steps is set to train_steps.
      """,
      default=1.0,
      type=float
  )
  args_parser.add_argument(
      '--train-method',
      help="""
        transfer, train from scratch, retrain
      """,
      #required=True,
      type=str,
      default= 'train_from_scratch',
      #choices=['transfer', 'retrain', 'train_from_scratch']
  )

  args_parser.add_argument(
      '--model-name',
      help="""
        Model name : 'MobileNet', 'VGG', 'ResNet'
      """,
      #required=True,
      default= 'Custom',
      type=str,
      #choices=['MobileNet', 'VGG', 'ResNet', 'Xception', 'Custom']
  )

  args_parser.add_argument(
      '--model-dir',
      help="""
        Model logs and weights directory
      """,
      default= '/home/vishal/Desktop/GIT/Sign_Language_Recognition/public/tf.record_as_input/trainer_module/trainer/models',
      #required=True,
      type=str
  )

  args_parser.add_argument(
      '--tfrecord-dir',
      help="""
        Tfrecord directory
      """,
      default='/home/vishal/Desktop/GIT/Sign_Language_Recognition/public/tf.record_as_input/trainer_module/trainer/tfrecord',
      type=str
  )

  args_parser.add_argument(
      '--bucket-name',
      help="""
        Bucket Name
      """,
      default='qommunicator',
      type=str
  )

  args_parser.add_argument(
      '--num-workers',
      help='Number of workers mentioned in config.yaml',
      default=6,
      type=int
  )

  args_parser.add_argument(
      '--strategy',
      help='Distribution strategy',
      #required=True,
      default= 'parameter_server',
      type=str,
      #choices=["mirror", "parameter_server", "None"]
  )

  return args_parser.parse_args()


def main():
  '''
  Main function to run task.py
  '''
  args = get_args()
  args = training.HParams(**args.__dict__)
  print('Batch Size', args.batch_size)
  print("before data")
  data = TfrecordInput(image_size=args.image_size,
                       num_classes=args.num_classes,
                       bucket_name=args.bucket_name,
                       batch_size=args.batch_size,
                       num_epochs=args.num_epochs,
                       data_dir=args.tfrecord_dir,
                       num_workers=args.num_workers)
  print("after data")
  data_valid = TfrecordInput(image_size=args.image_size,
                             num_classes=args.num_classes,
                             bucket_name=args.bucket_name,
                             batch_size=args.batch_size,
                             num_epochs=args.num_epochs,
                             data_dir=args.tfrecord_dir,
                             num_workers=args.num_workers,
                             mode='valid')
  model = ClassificationEstimator(num_class=args.num_classes,
                                  input_fn=data.data_input_fn(),
                                  batch_size=args.batch_size,
                                  model_name=args.model_name,
                                  learning_rate=args.learning_rate,
                                  train_steps=args.train_steps,
                                  eval_steps=args.eval_steps,
                                  strategy=args.strategy,
                                  eval_input_fn=data_valid.data_input_fn(),
                                  exporters=data.get_exporters(),
                                  eval_throttle_secs=args.eval_throttle_secs)

  print('*' * 80,
        'Running classification estimator over', args.num_classes,
        'classes\n', '*' * 80)
  model_dir = os.path.join(
      args.model_dir, json.loads(os.environ.get('TF_CONFIG', '{}'))\
          .get('task', {})\
              .get('trial', ''))
  if args.train_method == 'transfer':
    model.transfer(model_dir=model_dir)
  elif args.train_method == 'retrain':
    model.retrain(model_dir=model_dir)
  else:
    model.train_from_scratch(
        model_dir=model_dir)


if __name__ == "__main__":
  main()
