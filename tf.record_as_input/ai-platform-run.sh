# Google cloud arguments
REGION="us-east1"
BUCKET_NAME="qodelabs"
BUCKET_OBJECT="fashion_mnist"

# Module mode arguments
MODEL_NAME="VGG"
TRAIN_MODE="train_from_scratch"

# Jobs running arguments
TRAIN_STEPS=1200
EVAL_STEPS=10
BATCH_SIZE=16
LEARNING_RATE=0.0001
BATCH_SIZE_HP=( 8 16 32 )
LEARNING_RATE_HP=( 0.0001 0.0005 0.001 )
NUM_CLASSES=10
NUM_EPOCHS=2
IMAGE_SIZE=28
WORKERCOUNT=3
PARAMETERSERVERCOUNT=1
SAVE_TIME="False"
HYPERTUNING="False"        
MAX_TRIALS=4
EVAL_THROTTLE_SECS=100

TRAINER_PACKAGE_PATH="$(pwd)/trainer/"
now=$(date +"%Y%m%d_%H%M%S")
MAIN_TRAINER_MODULE="trainer.task"
DATA_PATH="$BUCKET_OBJECT/tfrecord/"
PACKAGE_STAGING_PATH="gs://$BUCKET_NAME/"
JOB_NAME="$BUCKET_NAME"_"$MODEL_NAME"_"$TRAIN_MODE"_"$now"
MODEL_DIR="gs://$BUCKET_NAME/$BUCKET_OBJECT/logs_$MODEL_NAME"_"$now/"
CONFIG="config.yaml"

STRATEGY=$(python3 create_config.py --train-method $TRAIN_MODE \
 --workercount $WORKERCOUNT \
 --parameterservercount $PARAMETERSERVERCOUNT \
 --hypertuning $HYPERTUNING \
 --model-name $MODEL_NAME \
 --save-time $SAVE_TIME \
 --max-trials $MAX_TRIALS \
 --batch-size ${BATCH_SIZE_HP[*]} \
 --learning-rate ${LEARNING_RATE_HP[*]})

echo "Using strategy :"$STRATEGY 

gcloud ai-platform jobs submit training $JOB_NAME \
    --staging-bucket $PACKAGE_STAGING_PATH \
    --runtime-version 1.14 \
    --python-version 3.5 \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region $REGION \
    --config $CONFIG \
    -- \
    --train-method $TRAIN_MODE \
    --model-name $MODEL_NAME \
    --bucket-name $BUCKET_NAME\
    --model-dir $MODEL_DIR \
    --tfrecord-dir $DATA_PATH \
    --train-steps $TRAIN_STEPS \
    --eval-steps $EVAL_STEPS \
    --num-epochs $NUM_EPOCHS \
    --num-classes $NUM_CLASSES \
    --image-size $IMAGE_SIZE \
    --num-workers $WORKERCOUNT \
    --strategy $STRATEGY \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --eval-throttle-secs $EVAL_THROTTLE_SECS 
