import os
import sys
import argparse
import torch
import numpy
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
from dataloader import LMDataSet
from utils import custom_load_model, noam_scheduler, LinearScheduler
from utils import show_bar, count_parameters, model_path
from models import LMBase

os.environ['MASTER_ADDR'] = 'localhost'  # Example IP address, replace with your master node's IP
os.environ['MASTER_PORT'] = '12340'        # Example port, choose an available port

english_vocab = [';', 
    'a', 'b', 'c', 'd', 'e',
    'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y',
    'z', ' ', '.', ',', '\n', '\t']
math_vocab = [';',
    '1', '2', '3', '4',
    '5', '6', '7', '8',
    '9', '0', '+', '-',
    '=', ' ', '\n']

class Tokenizer(object):
    def __init__(self):
        self.inverse_dict = math_vocab
        #self.inverse_dict = english_vocab
        self.vocab_dict = dict()
        for i, cha in enumerate(self.inverse_dict):
            self.vocab_dict[cha] = i

    def tokenize(self, input_str):
        output = []
        for cha in input_str.lower():
            output.append(self.vocab_dict[cha])
        return output

    def inverse_tokenize(self, tokens):
        output = ""
        for token in tokens:
            output += self.inverse_dict[token]
        return output

def main_epoch(rank, use_gpu, world_size,
        vocab_size, load_model_path, max_time_step, data, tokenizer):
    if use_gpu:
        torch.cuda.set_device(rank)  # Set the current GPU to be used
        device = torch.device(f'cuda:{rank}')
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        device = torch.device('cpu')
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Create model and move it to GPU with id `gpu`
    model = LMBase(vocab_size=vocab_size,
                hidden_size=1024,
                nhead=16,
                max_time_step=max_time_step,
                n_trn_block=12)

    model = model.to(device)

    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    # Example dataset and dataloader
    model = custom_load_model(model, f'{load_model_path}/model.pth')

    model.eval()
    tokens = torch.tensor(tokenizer.tokenize(data), dtype=torch.int64, device=device).unsqueeze(0)
    print("\n\nTHE OUTPUT SAMPLING:\n\n")

    T_setting_math = {0: 0.8, 13:0.01}
    T_setting_eng = {0: 1.0}
    l = 512

    outputs = model.module.inference_seg(tokens, l, T_default=0.30, T_setting=T_setting_math)
    outputs = tokenizer.inverse_tokenize(outputs[0].tolist())
    print(outputs[-l:])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--input_text', type=str)
    parser.add_argument('--max_time_step', type=int, default=1024)
    parser.add_argument('--vocab_size', type=int, default=32)
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_gpu else os.cpu_count()
    if(use_gpu):
        print("Use Parallel GPUs: %s" % world_size)
    else:
        print("Use Parallel CPUs: %s" % world_size)

    tokenizer = Tokenizer()
    with open(args.input_text, 'r') as f_reader:
        data = f_reader.read().strip()
        print("\nTHE INPUT IS:\n%s\n" % data)

    tokens = tokenizer.tokenize(data)

    mp.spawn(main_epoch,
             args=(use_gpu, world_size,
                    args.vocab_size,
                    args.load_path,
                    args.max_time_step, 
                    data, tokenizer),
             nprocs=world_size if use_gpu else min(world_size, 4),  # Limit CPU processes if desired
             join=True)
