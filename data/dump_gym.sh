#!/bin/bash

# Run the gen_gym_record.py script

ENV_NAME="LAKE" # Name of the gym environment, choices=['LAKE', 'LANDER'], default is LAKE,
SAVE_PATH="./gym_data" # Path to save the training data (without file extension)
POLICY_NAME="PPO" # Policy type, default is DQN, can be DQN, TD3, A2C, or PPO. Note, TD3 is for continous problem.
N_TOTAL_TIMESTEPS=20000 # Total number of epochs for training, default is 200000
N_TASK=1 # Total number of tasks for generating, default is 1. If env isn't random, set 1.
N_SEQ_LEN=4000 # Maximum number of actions per sequence, default is 4000.
N_WORKERS=10 # Number of parallel workers for segment data generation, default is 10
ENABLE_LOAD_MODEL="False" # Whether to load a pre-trained model, default is False
RANDOM_ENV="False" # Whether to use random environment, default is False

# Run the gen_gym_record.py script
python gen_gym_record.py \
    --env_name $ENV_NAME \
    --save_path $SAVE_PATH \
    --policy_name $POLICY_NAME \
    --n_total_timesteps $N_TOTAL_TIMESTEPS \
    --n_task $N_TASK \
    --n_seq_len $N_SEQ_LEN \
    --n_workers $N_WORKERS \
    --enable_load_model $ENABLE_LOAD_MODEL \
    --random_env $RANDOM_ENV