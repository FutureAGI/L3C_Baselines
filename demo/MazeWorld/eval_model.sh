scale=25
ext=eval.dyn.$scale
export CUDA_VISIBILE_DEVICES=0
nohup python evaluate.py config_maze.yaml \
	--configs \
		demo_config.run_model=1 \
		demo_config.run_rule=0 \
		demo_config.run_random=0 \
		demo_config.time_step=2048 \
		demo_config.model_config.load_model_path=checkpoints/pretrain_20240523_02/ \
		demo_config.read_task=/root/workspace/data/maze_evaluate_tasks/maze_"$scale".pkl \
		demo_config.output=./eval_results_"$scale"_standard \
		> log.$ext &

ext=eval.dyn.$scale.small
export CUDA_VISIBILE_DEVICES=1
nohup python evaluate.py config_maze_small.yaml \
	--configs \
		demo_config.run_model=1 \
		demo_config.run_rule=0 \
		demo_config.run_random=0 \
		demo_config.time_step=2048 \
		demo_config.model_config.load_model_path=checkpoints/pretrain_20240523_small/ \
		demo_config.read_task=/root/workspace/data/maze_evaluate_tasks/maze_"$scale".pkl \
		demo_config.output=./eval_results_"$scale"_small \
		> log.$ext &
