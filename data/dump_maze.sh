python gen_lm_data.py \
  --version v2 \
  --vocab_size 32,64,128 \
  --embedding_size 16,32 \
  --hidden_size 16,32,64 \
  --n_gram 2,3,4,5 \
  --sequence_length 8192 \
  --file_size 500 \
  --file_number 1000 \
  --workers 32 \
  --output_path /root/workspace/afs/lm_data_train/