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
os.environ['MASTER_PORT'] = '12342'        # Example port, choose an available port


def main_epoch(rank, use_gpu, world_size,
        batch_size, vocab_size, data_path, load_model_path, 
        max_time_step, test_time_step):
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
    dataset = LMDataSet(data_path, 500, verbose=True)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    all_length = len(dataloader)

    if(load_model_path is not None):
        model = custom_load_model(model, f'{load_model_path}/model.pth')

    model.eval()

    results = []
    for batch_idx, (feature, label) in enumerate(dataloader):
        feature = feature[:, :test_time_step].to(device)
        label = label[:, :test_time_step].to(device)
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
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_time_step', type=int, default=1024)
    parser.add_argument('--test_time_step', type=int, default=256)
    parser.add_argument('--vocab_size', type=int, default=64)
    parser.add_argument('--load_path', type=str, default=None)
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_gpu else os.cpu_count()
    if(use_gpu):
        print("Use Parallel GPUs: %s" % world_size)
    else:
        print("Use Parallel CPUs: %s" % world_size)

    mp.spawn(main_epoch,
             args=(use_gpu, world_size,
                    args.batch_size, 
                    args.vocab_size,
                    args.data_path,
                    args.load_path,
                    args.max_time_step, 
                    args.test_time_step),
             nprocs=world_size if use_gpu else min(world_size, 4),  # Limit CPU processes if desired
             join=True)
