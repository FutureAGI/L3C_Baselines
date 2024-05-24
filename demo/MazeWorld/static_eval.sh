length=512
export CUDA_VISIBLE_DEVICES=0
ext=sta."$length".txt
nohup python static_evaluate.py config_maze.yaml \
	--config test_config.segment_length=$length \
    > log.eval.$ext &

#export CUDA_VISIBLE_DEVICES=3
#ext=sta."$length".small.txt
#nohup python static_evaluate.py config_maze_small.yaml \
#	--config test_config.segment_length=$length \
#    > log.eval.$ext &
