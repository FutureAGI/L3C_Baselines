scale=15
ext=eval.dyn.$scale
export CUDA_VISIBILE_DEVICES=7
nohup python evaluate.py \
	--run_model 1 \
	--run_rule 0 \
	--load_path checkpoints/pretrain_20240509/ \
	--max_time_step 2048 \
	--test_time_step 2048 \
	--read_task /root/workspace/data/maze_evaluate_tasks/maze_"$scale".pkl \
	--output ./eval_results_$scale > log.$ext &
tail -f log.$ext
