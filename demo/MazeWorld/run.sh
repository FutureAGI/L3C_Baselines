export CUDA_VISIBLE_DEVICES=4,5
python train.py \
	--train_data_path /root/workspace/data/maze_data_train/ \
	--test_data_path /root/workspace/data/maze_data_demo/ \
	--max_time_step 2048 \
	--vae_batch_size 1 \
	--sequential_batch_size 1 \
	--train_time_step 128 \
	--eval_interval 1 \
	--lr 0.0005 \
	--max_epochs 30 \
	--vae_stop_epoch -1 \
	--main_start_epoch -1 \
	--load_path ./checkpoints/pretrain_20240428_02 \
	--save_path ./checkpoints/ 
