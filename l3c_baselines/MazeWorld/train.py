import argparse
import os
import torch
import numpy
from reader import MazeDataSet
from models import MazeModels
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

os.environ['MASTER_ADDR'] = 'localhost'  # Example IP address, replace with your master node's IP
os.environ['MASTER_PORT'] = '12345'        # Example port, choose an available port

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(rank, use_gpu, world_size, data_path, time_step, learning_rate):
    if use_gpu:
        torch.cuda.set_device(rank)  # Set the current GPU to be used
        device = torch.device(f'cuda:{rank}')
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        device = torch.device('cpu')
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    print("use_gpu", use_gpu, "rank:", rank, device)

    # Create model and move it to GPU with id `gpu`
    model = MazeModels(image_size=256, map_size=7, action_size=5)
    print("Number of parameters: ", count_parameters(model))
    model = model.to(device)
    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    # Example dataset and dataloader
    dataset = MazeDataSet(data_path, time_step)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Example training loop
    total_iteration = 0
    for obs,acts,rews,maps in dataloader:
        obs = obs.to(device)
        acts = acts.to(device)
        rews = rews.to(device)
        maps = maps.to(device)
        print(obs.shape, maps.shape)
        obs = obs.permute(0, 1, 4, 2, 3)
        maps = maps.permute(0, 1, 4, 2, 3)
        lmse1, lmse2, lce = model.module.train_loss(obs, acts, rews, maps)
        loss = lmse1 + lmse2 + lce

        try:
            optimizer.zero_grad()
            loss.backward()
            prt_loss = (lmse1.detach().numpy(), lmse2.detach().numpy(), lce.detach().numpy())
            total_iteration += 1
            print("Iteration: %s; Current File: %s; Loss: Future Prediction MSE: %s, Map Prediction MSE: %s, Action Cross Entropy: %s" % prt_loss)
            sys.stdout.flush()
            optimizer.step()
        except RuntimeError as e:
            print("RuntimeError during backward pass:", e)
            # Additional debugging steps or handling of the error
            break  # Exit the loop or handle the error appropriately

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--time_step', type=int, default=64)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_gpu else os.cpu_count()
    if(use_gpu):
        print("Use Parallel GPUs: %s" % world_size)
    else:
        print("Use Parallel CPUs: %s" % world_size)

    mp.spawn(train_epoch,
             args=(use_gpu, world_size, args.data_path, args.time_step, args.lr),
             nprocs=world_size if use_gpu else min(world_size, 4),  # Limit CPU processes if desired
             join=True)
