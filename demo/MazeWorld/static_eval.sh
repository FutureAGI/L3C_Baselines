export CUDA_VISIBLE_DEVICES=7
nohup python static_evaluate.py --test_data_path /root/workspace/data/maze_data_demo/ --test_batch_size 1 --load_path ./checkpoints/pretrain_20240429_01/ > log.eval.sta &
tail -f log.eval.sta
