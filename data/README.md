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
  --reference_policy_config 0.50 \
  --behavior_policy_config 0.15,0.50 0.20,0.60 0.30,0.75 \
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

## Data Collection Strategy
In Maze World, we use different behavior policy and reference policy to facilitate the performance of imitation learning.
Inspired by (Lin et al., 2023; Zisman et al., 2023; Lee et al., 2024), it is possible to distill in-context reinforcement learning algorithm by imitation learning and noise distilling.

Use "--reference_policy_config" to specify the $p(STM->LTM)$ of the reference policy, e.g., "--reference_policy_config 0.50" represents $p(STM->LTM) = 50\%$.

Use "--behavior_policy_config" to specify the $p(STM->LTM)$ and the $epsilon$ combination of the behavior policy, the option support multiple choices, e.g., "--behavior_policy_config 0.15,0.50 0.20,0.60 0.30,0.75" represents $p(STM->LTM) = 15\%, 20\%, 30\%$ and $epsilon = 50\%, 60\%, 75\%$
