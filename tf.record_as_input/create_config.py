'''
Creates a config.yaml file
'''
import yaml
import argparse


def get_args():
  '''
  Define the task arguments with the default values.
Returns:
  experiment parameters
  '''
  args_parser = argparse.ArgumentParser()
  args_parser.add_argument(
      '--train-method',
      help="""
        transfer, train from scratch, retrain
      """,
      required=True,
      type=str,
      choices=['transfer', 'retrain', 'train_from_scratch']
  )
  args_parser.add_argument(
      '--batch-size',
      nargs='+',
      help="batch size as list")
  args_parser.add_argument(
      '--learning-rate',
      nargs='+',
      type=float,
      help="list of learning rate")

  args_parser.add_argument(
      '--workercount',
      help='Number of workers mentioned in config.yaml',
      default=6,
      type=int
  )
  args_parser.add_argument(
      '--parameterservercount',
      help='Number of parameterserver worker',
      default=3,
      type=int
  )
  args_parser.add_argument(
      '--max-trials',
      help='maximum number of trials in hyperparametertuning',
      default=3,
      type=int
  )
  args_parser.add_argument(
      '--model-name',
      help="""
        Model name : 'MobileNet', 'VGG', 'ResNet', 'Xception', 'Custom'
      """,
      required=True,
      type=str,
      choices=['MobileNet', 'VGG', 'ResNet', 'Xception', 'Custom']
  )
  args_parser.add_argument(
      '--save-time',
      help="""
        True, if wish to save time but more expensive
      """,
      required=True,
      type=str,
      choices=["True", "False"]
  )
  args_parser.add_argument(
      '--hypertuning',
      help="""
        True, if hypertuning needs to be done
      """,
      required=True,
      type=str,
      choices=["True", "False"]
  )
  return args_parser.parse_args()


class Createconfig():
  '''
  Class to create config.yaml
  '''

  def __init__(self,
               model_name,
               save_time,
               train_method,
               workercount,
               parameterservercount,
               max_trials,
               batch_size,
               learning_rate):
    """
    Initialization
    """
    self.model_name = model_name
    self.save_time = save_time
    self.train_method = train_method
    self.workercount = workercount
    self.parameterservercount = parameterservercount
    self.max_trials = max_trials
    self.batch_size = batch_size
    self.learning_rate = learning_rate

  @staticmethod
  def standard_config():
    '''
    standard config file
    '''
    config = {
        'trainingInput': {
            'scaleTier': 'CUSTOM',
            'masterType': 'standard_p100'
            }}
    return config

  def hyperparametertune(self, config):
    '''
    Hyperparamter tuning
    '''
    config['trainingInput']['hyperparameters'] = {
                'goal': 'MAXIMIZE',
                'hyperparameterMetricTag': 'accuracy',
                'maxTrials': self.max_trials,
                'maxParallelTrials': 3,
                'enableTrialEarlyStopping': False,
                'params': [{
                    'parameterName': 'batch-size',
                    'type': 'DISCRETE',
                    'discreteValues': self.batch_size}, {
                        'parameterName': 'learning-rate',
                        'type': 'DISCRETE',
                        'discreteValues': self.learning_rate}]}
    return config

  def decide_strategy(self):
    '''
    Deciding the GPU distribution strategy and config.yaml
    '''
    config = self.standard_config()
    if self.train_method == "transfer":
      strategy = "None"
    else:
      if self.model_name == "ResNet":
        config['trainingInput']['masterType'] = 'complex_model_m_gpu'
        strategy = "mirror"
      else:
        if self.save_time == "True":
          config['trainingInput']['masterType'] = 'standard_gpu'
          config['trainingInput']['workerType'] = 'standard_gpu'
          config['trainingInput']['parameterServerType'] = 'large_model'
          config['trainingInput']['workerCount'] = self.workercount
          config['trainingInput']['parameterServerCount'] = \
              self.parameterservercount
          strategy = "parameter_server"
        else:
          config['trainingInput']['masterType'] = 'complex_model_m_gpu'
          strategy = "mirror"
    return strategy, config


def main():
  '''
  Main function to create config file
  '''
  args = get_args()
  create_config = Createconfig(model_name=args.model_name,
                               save_time=args.save_time,
                               train_method=args.train_method,
                               workercount=args.workercount,
                               parameterservercount=args.parameterservercount,
                               max_trials=args.max_trials,
                               batch_size=args.batch_size,
                               learning_rate=args.learning_rate)
  strategy, config = create_config.decide_strategy()
  if args.hypertuning == "True":
    config = create_config.hyperparametertune(config)
  with open('config.yaml', 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)
  print(strategy)

if __name__ == "__main__":
  main()
