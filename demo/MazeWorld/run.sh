export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
python train.py \
	--train_data_path /root/workspace/L3C_baselines.bak/data/maze_train_data/ \
	--test_data_path /root/workspace/L3C_baselines.bak/data/maze_data/ \
	--max_time_step 1024 \
	--vae_batch_size 1 \
	--sequential_batch_size 3 \
	--train_time_step 256 \
	--eval_interval 1 \
	--lr 0.0005 \
	--max_epochs 30 \
	--vae_stop_epoch -1 \
	--main_start_epoch -1 \
	--load_path ./checkpoints/08 \
	--save_path ./checkpoints/ 
