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
from utils import Configure
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
    def __init__(self, vocab):
        self.inverse_dict = vocab
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

def main_epoch(rank, use_gpu, world_size, config, datas):
    if use_gpu:
        torch.cuda.set_device(rank)  # Set the current GPU to be used
        device = torch.device(f'cuda:{rank}')
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        device = torch.device('cpu')
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if(config.demo_config.vocab.lower() == "english"):
        print("English Vocabulary Used")
        T_setting = {0: 1.0}
        vocab = english_vocab
    elif(config.demo_config.vocab.lower() == "math"):
        print("Math Vocabulary Used")
        T_setting = {0: 0.8, 13:0.01}
        vocab = math_vocab
    tokenizer = Tokenizer(vocab)
        

    # Create model and move it to GPU with id `gpu`
    model = LMBase(config.model_config)
    model = model.to(device)
    load_model_path = config.demo_config.load_model_path

    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    # Example dataset and dataloader
    model = custom_load_model(model, f'{load_model_path}/model.pth', strict_check=False)

    print("\n\nTHE OUTPUT SAMPLING:\n\n")
    for data in datas:
        model.eval()
        tokens = torch.tensor(tokenizer.tokenize(data), dtype=torch.int64, device=device).unsqueeze(0)
        l = 512
        outputs = model.module.inference_seg(tokens, l, T_default=0.30, T_setting=T_setting)
        outputs = tokenizer.inverse_tokenize(outputs[0].tolist())
        print(outputs[-l:])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration', type=str, help="YAML configuration file")
    parser.add_argument('--configs', nargs='*', help="List of all configurations, overwrite configuration file: eg. train_config.batch_size=16 test_config.xxx=...")
    parser.add_argument('--data', type=str, help="Input contexts")
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_gpu else os.cpu_count()
    if(use_gpu):
        print("Use Parallel GPUs: %s" % world_size)
    else:
        print("Use Parallel CPUs: %s" % world_size)

    config = Configure()
    config.from_yaml(args.configuration)

    # Get the dictionary of attributes
    if args.configs:
        for pair in args.configs:
            key, value = pair.split('=')
            config.set_value(key, value)
            print(f"Rewriting configurations from args: {key} to {value}")
    print("Final configuration:\n", config)
    os.environ['MASTER_PORT'] = config.demo_config.master_port        # Example port, choose an available port

    with open(args.data, 'r') as f:
        data = f.readlines()
        mp.spawn(main_epoch,
                 args=(use_gpu, world_size, config, data),
                 nprocs=world_size if use_gpu else min(world_size, 4),  # Limit CPU processes if desired
                 join=True)
