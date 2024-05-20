if [[ $# -lt 1 ]]; then
	echo "Usage: $1 output_directory"
	exit 1
fi
echo "Output to $1"
python gen_metalang.py \
  --vocab_size 16 \
  --embedding_size 16,32 \
  --hidden_size 16,32,64 \
  --n_gram 3,4,5,6,7,8 \
  --sequence_length 4096 \
  --file_size 500 \
  --file_number 8 \
  --workers 8 \
  --output_path $1
