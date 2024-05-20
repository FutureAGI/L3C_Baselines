#!/bin/bash
#export CUDA_VISIBLE_DEVICES=4,5
export CUDA_VISIBLE_DEVICES=1,4,5
python train.py config_maze.yaml 
