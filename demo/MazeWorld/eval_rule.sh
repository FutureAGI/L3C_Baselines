scale=15
mem_kr=1
nohup python evaluate.py config_maze.yaml \
	--configs \
		demo_config.run_model=0 \
		demo_config.run_rule=0 \
		demo_config.run_random=1 \
        demo_config.time_step=2048 \
        demo_config.rule_config.mem_kr=${mem_kr} \
        demo_config.output=./eval_results_'$scale'_mem_'$mem_kr' \
        demo_config.read_task=/root/workspace/data/maze_evaluate_tasks/maze_'$scale'.pkl \
        > log.'$scale'.mem.'$memkr' &
