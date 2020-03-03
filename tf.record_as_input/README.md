# Trainer package
Training module for image classification

## Usage
The script for running the job on google AI-Platform is provided in ` ai-platform-run.sh`. Provide the values for following variables **referring** to script:
```bash
# Google cloud arguments
REGION="your-region"
BUCKET_NAME="your-bucket-name"
BUCKET_OBJECT="folder-name"         # Folder name contained in the given bucket 

# Module mode arguments
MODEL_NAME="model-name"             # VGG | MobileNet | ResNet  | Xception | Custom (any one)
TRAIN_MODE="mode"                   # train_from_scratch | transfer | retrain (any one)

# Jobs running arguments
TRAIN_STEPS=<int>                   # number of steps
EVAL_STEPS=<int>                    # number of steps before evaluation
BATCH_SIZE=<int>                    # batch size
NUM_EPOCHS=<int>                    # number of epochs
LEARNING_RATE=<float>               # learning rate for training
NUM_CLASSES=<int>                   # number of classes to predict
NUM_EPOCHS=<int>                    # num of epochs while training
IMAGE_SIZE=<int>                    # input image size
WORKERCOUNT=<int>                   # number of worker GPU
SAVE_TIME=<str>                     # "True" if you want to complete training quickly, "False" if you need to save cost
HYPERTUNING=<str>                   # "True" if you want to do hypertuning
MAX_TRIALS=<int>                    # Number of maximum trials during hypertuning
BATCH_SIZE_HP=<tuple>               # space separated integral batch size encosed in round bracket in ascending order
LEARNING_RATE_HP=<tuple>            # space separated float learning rate encosed in round bracket in ascending order
PARAMETERSERVERCOUNT=<int>          # Number of Parameterserver to be used in Parameter server strategy
```

To run the script type the following commands:
```bash
cd trainer_module
bash ai-platform-run.sh 
```

