#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py config_maze_small.yaml 
