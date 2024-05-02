export CUDA_VISIBLE_DEVICES=7
nohup python static_evaluate.py --test_data_path /root/workspace/data/maze_test.15/ --test_batch_size 1 --load_path ./checkpoints/pretrain_20240430/ > log.eval.sta.15.txt &
tail -f log.eval.sta.15.txt
