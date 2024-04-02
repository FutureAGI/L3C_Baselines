export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
nohup python train.py \
	--train_data_path /root/workspace/L3C_baselines.bak/data/maze_test_data/ \
	--test_data_path /root/workspace/L3C_baselines.bak/data/maze_data/ \
	--max_time_step 1024 \
	--train_time_step 128 \
	--vae_batch_size 2 \
	--sequential_batch_size 4 \
	--train_time_step 128 \
	--eval_interval 1 \
	--lr 0.001 \
	--max_epochs 30 \
	--vae_stop_epoch 10 \
	--main_start_epoch 8 \
	--save_path ./checkpoints/ > log.train &

	#--load_path ./checkpoints/30 \
