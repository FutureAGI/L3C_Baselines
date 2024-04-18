#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
python train.py \
    --train_data_path /root/workspace/data/lm_data_train \
    --test_data_path /root/workspace/data/lm_data_demo \
    --train_time_step 1024 \
    --max_time_step 1024 \
    --vocab_size 16 \
    --lr 1.0e-3 \
    --batch_size 1
