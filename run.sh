#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4
#mode=TRN
mode=RMT
#mode=LSTM
data_format="v2.hid32.voc8.emb8"
#data_format="v1.elen16.voc16.en1"
nohup /usr/bin/python3.9 -m paddle.distributed.launch --gpus=$gpus --log_dir logs.$mode train.py \
    --version v2 \
    --vocab_size 8 \
    --embedding_size 8 \
    --hidden_size 32 \
    --elements_length 16 \
    --elements_number 1 \
    --error_rate 0.01 \
    --sequence_length 512 \
    --train_segment 256 \
    --eval_segment 256 \
    --detach_segments 4 \
    --batch_size 16 \
    --model_type $mode \
    --model_layer_num 12 \
    --model_hidden_size 768 \
    --model_head_num 16 \
    --model_memory_size 128 \
    --opt_learning_rate 0.0001 \
    --opt_warmup_steps 2000 \
    --evaluation_data_path data/data.$data_format.txt \
    --model_load_path checkpoint.RMT/epoch-200 \
    --train_warmup_steps 0 \
    --train_intermediate_steps 128 > res.$data_format.$mode.txt &
