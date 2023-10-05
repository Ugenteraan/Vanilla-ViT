'''Configurations.
'''

#GENERAL CONFIGURATIONS
DEVICE='gpu'
MODEL_SAVE_FOLDER='./artifacts/'
GRAPH_SAVE_FOLDER='./graphs/'

#DATASET CONFIGURATIONS
BATCH_SIZE=2
IMAGE_WIDTH=224
IMAGE_HEIGHT=224
IMAGE_CHANNEL=1
SHUFFLE=False
PATCH_SIZE=16

#MODEL CONFIGURATIONS
TRANSFORMER_NETWORK_DEPTH=8
NUM_CLASSES=205
PROJECTION_DIM_KEYS=512
PROJECTION_DIM_VALUES=512
NUM_HEADS=8
ATTN_DROPOUT_PROB=0
FEEDFORWARD_PROJECTION_DIM=1024
FEEDFORWARD_DROPOUT_PROB=0

#TRAINING CONFIGURATIONS
TRAIN_EPOCH=100
LEARNING_RATE=0.001
SCHEDULER_GAMMA=0.7
SCHEDULER_STEP_SIZE=1




