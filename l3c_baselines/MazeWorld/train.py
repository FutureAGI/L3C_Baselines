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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(rank, use_gpu, world_size, model, dataset, learning_rate):
    if use_gpu:
        torch.cuda.set_device(rank)  # Set the current GPU to be used
        device = torch.device(f'cuda:{rank}')
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        device = torch.device('cpu')
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Create model and move it to GPU with id `gpu`
    model = model.to(device)
    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    # Example dataset and dataloader
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=ngpus_per_node, rank=gpu)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Example training loop
    total_iteration = 0
    for obs,acts,rews,maps in dataloader:
        obs = obs.to(gpu)
        acts = acts.to(gpu)
        rews = rews.to(gpu)
        maps = maps.to(gpu)
        obs = obs.permute(0, 3, 1, 2)
        maps = maps.permute(0, 3, 1, 2)
        loss = model.transfer_loss(obs, acts, rews, maps)

        try:
            optimizer.zero_grad()
            loss.backward()
            prt_loss = loss.detach().numpy()
            total_iteration += 1
            print("Iteration: %s; Current File: %s; Loss: %s" % (total_iteration, file_name, prt_loss))
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
    file_names = os.listdir(args.data_path)
    model = MazeModels(image_size=256, map_size=7, action_size=5)
    dataset = MazeDataSet(args.data_path, args.time_step)
    print("Number of parameters: ", count_parameters(model))

    use_gpu = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_gpu else os.cpu_count()

    mp.spawn(main_worker,
             args=(rank, use_gpu, world_size, model, dataset, learning_rate, args.lr),
             nprocs=world_size if use_gpu else min(world_size, 4),  # Limit CPU processes if desired
             join=True)
