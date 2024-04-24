# L3C_Baselines/data
Generate the benchmark data for l3c baselines

## Usage
```shell
python gen_maze_record.py 
  --output_path OUTPUT_PATH
  --task_type TASK_TYPE
  --maze_type MAZE_TYPE
  --max_steps MAX_STEPS
  --n n1,n2,n3
  --density d1,d2,d3
  --n_landmarks l1,l2,l3
  --epochs EPOCHS
  --workers WORKERS
```

## Output
```shell
python gen_metalang.py 
	--vocab_size 64
	--embedding_size 16
	--n_gram 3
	--file_size 1000
	--file_number 100
	--output_path PATH_TO_DIR
	--sequence_length 2048 
	--file_number 16
	--workers 4
```
