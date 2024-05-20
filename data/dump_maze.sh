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
  --memory_keep 1.0 \
  --landmarks 10 \
  --scale 15,25,35 \
  --epochs 64 \
  --workers 32
