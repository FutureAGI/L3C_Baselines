scale=15
ext=eval.dyn.$scale
export CUDA_VISIBILE_DEVICES=7
nohup python evaluate.py config_maze.yaml \
	--configs \
		demo_config.run_model=1 \
		demo_config.run_rule=0 \
		demo_config.run_random=0 \
		demo_config.time_step=2048 \
		demo_config.model_config.load_model_path=checkpoints/pretrain_20240504/ \
		demo_config.read_task=/root/workspace/data/maze_evaluate_tasks/maze_"$scale".pkl \
		demo_config.output=./eval_results_"$scale"_old \
		> log.$ext.old &
tail -f log.$ext.old
