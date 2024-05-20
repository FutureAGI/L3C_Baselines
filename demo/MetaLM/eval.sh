export CUDA_VISIBLE_DEVICES=5
ngram=3
file=res.natural.tiny
python evaluate.py config_mlm.yaml \
    --configs test_config.data_path=/root/workspace/data/natural_lang \
              train_config.load_model_path=./checkpoints/pretrain-tiny-20240508 \
              model_config.max_time_step=4096 \
              model_config.vocab_size=32 \
			  test_config.time_step=4096 \
              train_config.batch_size=8 > $file &
tail -f $file
