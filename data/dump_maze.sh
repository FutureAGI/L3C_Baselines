if [[ $# -lt 1 ]]; then
	echo "Usage: $1 output_directory"
	exit 1
fi
echo "Output to $1"
python gen_maze_record.py \
  --output_path $1 \
  --task_type NAVIGATION \
  --maze_type Discrete3D \
  --max_steps 2048 \
  --label_mem_kr 0.50 \
  --behavior_mem_kr 0.15,0.30,0.50,0.75 \
  --behavior_noise 0.0,0.20,0.40,0.60,0.80 \
  --start_index 1280 \
  --landmarks 10 \
  --scale 15,25,35 \
  --epochs 48 \
  --workers 48
