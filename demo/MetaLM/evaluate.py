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

os.environ['MASTER_ADDR'] = 'localhost' 

def main_epoch(rank, use_gpu, world_size, config):
    if use_gpu:
        torch.cuda.set_device(rank)  # Set the current GPU to be used
        device = torch.device(f'cuda:{rank}')
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        device = torch.device('cpu')
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Create model and move it to GPU with id `gpu`
    model = LMBase(config.model_config)
    model = model.to(device)

    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    # Example dataset and dataloader
    test_config = config.test_config
    dataset = LMDataSet(test_config.data_path, test_config.file_size, verbose=True)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=test_config.batch_size, sampler=sampler)
    load_model_path = config.test_config.load_model_path
    all_length = len(dataloader)

    model = custom_load_model(model, f'{load_model_path}/model.pth', strict_check=False)

    model.eval()

    results = []
    time_step = test_config.time_step
    for batch_idx, (feature, label) in enumerate(dataloader):
        feature = feature[:, :time_step].to(device)
        label = label[:, :time_step].to(device)
        #with autocast():
        with torch.no_grad():
            loss = model.module.perplexity_array(feature, label)
        results.append(loss)
        show_bar((batch_idx + 1) / all_length, 100)
        sys.stdout.flush()
    results=torch.mean(torch.stack(results, dim=0), dim=0).cpu().tolist()

    print('\n'.join(map(str, results)))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration', type=str, help="YAML configuration file")
    parser.add_argument('--configs', nargs='*', help="List of all configurations, overwrite configuration file: eg. train_config.batch_size=16 test_config.xxx=...")
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
    os.environ['MASTER_PORT'] = config.test_config.master_port        # Example port, choose an available port

    mp.spawn(main_epoch,
             args=(use_gpu, world_size, config),
             nprocs=world_size if use_gpu else min(world_size, 4),  # Limit CPU processes if desired
             join=True)
