#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
mode=trnxl
#mode=lstm
gpus=6,7
#gpus=4,5
data_format="512.64.32"
nohup /usr/bin/python3.9 -m paddle.distributed.launch --gpus=$gpus --log_dir logs.$mode train.py data.mlm.$data_format.2000.dat > res.$mode.$data_format.dat &
