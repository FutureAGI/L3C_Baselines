#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#mode=TransformerXL
mode=Plasmer
#mode=lstm
#gpus=6,7
gpus=4,5,6,7
#data_format="v1.ele64_10"
data_format="v2.hid16"
nohup /usr/bin/python3.9 -m paddle.distributed.launch --gpus=$gpus --log_dir logs.$mode train.py \
    --version v2 \
    --vocab_size 64 \
    --embedding_size 4 \
    --hidden_size 16 \
    --elements_length 64 \
    --elements_number 10 \
    --error_rate 0.10 \
    --sequence_length 8192 \
    --train_segment 256 \
    --eval_segment 256 \
    --detach_segments 4 \
    --batch_size 16 \
    --model_type $mode \
    --model_layer_num 8 \
    --model_hidden_size 512 \
    --model_head_num 16 \
    --evaluation_data_path data/data.$data_format.txt \
    > res.$data_format.$mode.txt &
