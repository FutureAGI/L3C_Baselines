python -m l3c.metalm.data_generator --version v1 --vocab_size 256 --elements_length 64 --elements_number 10 --error_rate 0.05 --sequence_length 16384 --samples 2000 --output data.v1.ele64_10.txt
python -m l3c.metalm.data_generator --version v2 --vocab_size 256 --embedding_size 16 --hidden_size 16 --sequence_length 16384 --samples 2000 --output data.v2.hid16.txt
