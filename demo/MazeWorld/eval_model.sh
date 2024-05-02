ext=eval.dyn.15
export CUDA_VISIBILE_DEVICES=5
nohup python evaluate.py \
	--run_model 1 \
	--run_rule 0 \
	--load_path checkpoints/pretrain_20240501_01/ \
	--max_time_step 2048 \
	--test_time_step 2048  \
	--read_task /root/workspace/data/maze_evaluate_tasks/maze_15.pkl \
	--output ./eval_results_15 > log.$ext &
tail -f log.$ext
