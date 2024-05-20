#!/bin/bash
#export CUDA_VISIBLE_DEVICES=4,5
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py config_mlm.yaml \
    --configs train_config.data_path=/root/workspace/data/lm_data_train.32 \
              test_config.data_path=/root/workspace/data/lm_data_test.32 \
              train_config.time_step=4000 \
              model_config.max_time_step=4096 \
              model_config.vocab_size=32 \
              train_config.learning_rate=1.0e-3 \
              train_config.batch_size=2
