import os
import sys
import argparse
import torch
import numpy
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
from dataloader import MazeDataSet, PrefetchDataLoader
from utils import custom_load_model, noam_scheduler, LinearScheduler
from utils import show_bar, count_parameters, check_model_validity, model_path
from models import MazeModelBase1, MazeModelBase2

os.environ['MASTER_ADDR'] = 'localhost'  # Example IP address, replace with your master node's IP
os.environ['MASTER_PORT'] = '12345'        # Example port, choose an available port


def main_epoch(rank, use_gpu, world_size, 
        test_batch_size, test_data_path, 
        load_model_path,
        max_time_step, 
        test_time_step, 
        main_rank):
    if use_gpu:
        torch.cuda.set_device(rank)  # Set the current GPU to be used
        device = torch.device(f'cuda:{rank}')
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        device = torch.device('cpu')
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if(main_rank is None):
        main = False
    elif(main_rank == "all" or main_rank == rank):
        main = True
    else:
        main = False

    if(main):
        print("Main gpu", use_gpu, "rank:", rank, device)

    # Create model and move it to GPU with id `gpu`
    model = MazeModelBase2(image_size=128, map_size=7, action_size=5, max_time_step=max_time_step)
    if(main):
        print("Number of parameters: ", count_parameters(model))
        print("Number of parameters decision transformer: ", count_parameters(model.decformer))
        print("Number of parameters action decoder: ", count_parameters(model.act_decoder))
        print("Number of parameters encoder: ", count_parameters(model.encoder))
        print("Number of parameters decoder: ", count_parameters(model.decoder))

    model = model.to(device)

    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    test_dataset = MazeDataSet(test_data_path, test_time_step, verbose=main)
    test_dataloader = PrefetchDataLoader(test_dataset, batch_size=test_batch_size)

    model = custom_load_model(model, f'{load_model_path}/model.pth', black_list={})

    # Perform the first evaluation
    test_epoch(rank, use_gpu, world_size, test_dataloader, model, main, device, 0)

def test_epoch(rank, use_gpu, world_size, test_dataloader, model, main, device, epoch_id):
    # Example training loop
    results = []
    all_length = len(test_dataloader)

    if(main):
        print("[EVALUATION] Epochs: %s..." % epoch_id)
    lzs = []
    lacts = []
    lrecs = 0.0
    cnts = 0
    for batch_idx, batch in enumerate(test_dataloader):
        obs,acts,rews,maps = batch
        obs = obs.to(device)
        acts = acts.to(device)
        obs = obs.permute(0, 1, 4, 2, 3)
        length = acts.shape[0] * acts.shape[1]
        with torch.no_grad():
            lrec = model.module.vae_loss(obs)
            lz, lact, cnt = model.module.sequential_loss_with_decoding(obs, acts, reduce=None)

        lrecs += lrec.cpu() * cnt
        cnts += cnt
        lzs.append(lz)
        lacts.append(lact)

        if(main):
            show_bar((batch_idx + 1) / all_length, 100)
            sys.stdout.flush()

    lrecs/=cnts
    lrecs = lrecs.item()
    lzs=torch.mean(torch.stack(lzs, dim=0), dim=0).cpu().tolist()
    lacts=torch.mean(torch.stack(lacts, dim=0), dim=0).cpu().tolist()

    if(main):
        print("reconstruction loss", lrecs)
        print("prediction loss:\n\n")
        print("\n".join(map(str, lzs)))
        print("\n\naction loss:\n\n")
        print("\n".join(map(str, lacts)))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--max_time_step', type=int, default=2048)
    parser.add_argument('--test_time_step', type=int, default=2048)
    parser.add_argument('--load_path', type=str)
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_gpu else os.cpu_count()
    if(use_gpu):
        print("Use Parallel GPUs: %s" % world_size)
    else:
        print("Use Parallel CPUs: %s" % world_size)

    mp.spawn(main_epoch,
             args=(use_gpu, world_size, 
                    args.test_batch_size,
                    args.test_data_path,
                    args.load_path, 
                    args.max_time_step, 
                    args.test_time_step, 0),
             nprocs=world_size if use_gpu else min(world_size, 4),  # Limit CPU processes if desired
             join=True)
