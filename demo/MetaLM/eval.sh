export CUDA_VISIBLE_DEVICES=5
python evaluate.py \
	--data_path ../../../data/lm_data_demo/ \
	--load_path ./model/01/ \
	--vocab_size 16 \
	--batch_size 128 \
	--max_time_step 4096 \
	--test_time_step 1024
