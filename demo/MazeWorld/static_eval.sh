export CUDA_VISIBLE_DEVICES=5
length=2048
ext=sta."$length".txt
nohup python static_evaluate.py --test_data_path /root/workspace/data/maze_test.all/ --test_batch_size 1 --segment_length $length --load_path ./checkpoints/pretrain_20240509/ > log.eval.$ext &
tail -f log.eval.$ext
