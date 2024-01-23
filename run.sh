#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4
#mode=TRN
mode=RMT2
#mode=LSTM
data_format="v2.hid32.voc64.emb8.ng2"
#data_format="v1.elen16.voc16.en1"
nohup /usr/bin/python3.9 -m paddle.distributed.launch --gpus=$gpus --log_dir logs.$mode train.py \
    --version v2 \
    --vocab_size 64 \
    --embedding_size 8 \
    --hidden_size 32 \
    --n_gram 2 \
    --elements_length 16 \
    --elements_number 1 \
    --error_rate 0.20 \
    --sequence_length 512 \
    --train_segment 64 \
    --eval_segment 64 \
    --detach_segments 8 \
    --batch_size 16 \
    --model_type $mode \
    --model_layer_num 12 \
    --model_hidden_size 768 \
    --model_head_num 16 \
    --model_memory_size 64 \
    --opt_learning_rate 0.0001 \
    --opt_warmup_steps 2000 \
    --evaluation_data_path data/data.$data_format.txt \
    --model_load_path checkpoint.RMT2/epoch-200 \
    --train_max_epochs 200 \
    --train_epoch_max_iterations 500 \
    --train_warmup_steps 0 \
    --train_intermediate_steps 128 > res.$data_format.$mode.txt &

