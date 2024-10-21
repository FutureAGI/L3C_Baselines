#!/bin/bash

# Run the gen_gym_record.py script

ENV_NAME="LAKE" # Name of the gym environment, default is LAKE,
SAVE_PATH="./gym_data" # Path to save the training data (without file extension)
POLICY_NAME="A2C" # Policy type, default is DQN, can be DQN, TD3, A2C, or PPO. Note, TD3 is for continous problem.
N_TOTAL_TIMESTEPS=20000 # Total number of epochs for training, default is 200000
N_TASK=1000 # Total number of tasks for generating, default is 1000
N_MAX_TRY=100 # Maximum number of actions per epoch, default is 100
N_WORKERS=10 # Number of parallel workers for training, default is 1
# ENABLE_LOAD_MODEL=False # Whether to load a pre-trained model, default is False

# Run the gen_gym_record.py script
python gen_gym_record.py \
    --env_name $ENV_NAME \
    --save_path $SAVE_PATH \
    --policy_name $POLICY_NAME \
    --n_total_timesteps $N_TOTAL_TIMESTEPS \
    --n_task $N_TASK \
    --n_max_try $N_MAX_TRY \
    --n_workers $N_WORKERS \
    --enable_load_model False