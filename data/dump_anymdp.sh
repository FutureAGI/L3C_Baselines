 python3.8 ./gen_anymdp_record.py \
	 --output_path [OUTPUT_PATH] \
	 --task_source FILE \
	 --task_file [TASK_FILE_PATH] \
	 --max_steps 16000 \
	 --epochs 256 \
	 --workers 64 \
	 --behavior_noise_config 1.0,-1.0 1.0,-0.75 1.0,-0.5 1.0,-0.25 1.0,0.0 1.0,0.5 1.0,0.75 \
	 --behavior_config 1.0,0.0 0.5,0.5 0.0,1.0 \
	 --reference_config 1.0,0.0 0.5,0.5 0.2,0.8
