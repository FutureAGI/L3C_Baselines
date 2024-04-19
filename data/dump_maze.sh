  --output_path /root/workspace/afs/lm_data_train/

python gen_maze_record.py \
  --output_path /root/workspace/ \
  --task_type NAVIGATION \
  --maze_type Discrete3D \
  --max_steps 2048 \
  --n 9,15,21,25 \
  --epochs 1000 \
  --workers 32
