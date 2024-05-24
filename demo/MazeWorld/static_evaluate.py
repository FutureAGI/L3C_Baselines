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
from utils import Configure
from models import MazeModelBase


os.environ['MASTER_ADDR'] = 'localhost'  # Example IP address, replace with your master node's IP

def main_epoch(rank, use_gpu, world_size, config, main_rank):
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
    model = MazeModelBase(config.model_config)
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

    test_config = config.test_config
    dataset = MazeDataSet(test_config.data_path, test_config.time_step, verbose=main)
    dataloader = PrefetchDataLoader(dataset, batch_size=test_config.batch_size)

    model = custom_load_model(model, f'{test_config.load_model_path}/model.pth', black_list={}, strict_check=True)

    # Perform the first evaluation
    test_time_step = test_config.time_step
    seg_len = test_config.segment_length
    seg_num = test_time_step // seg_len
    lobss = []
    lrecs = []
    lacts = []
    for seg_id in range(seg_num):
        print(f"\nrun_segment {seg_id + 1}/{seg_num}...\n\n")
        seg_b = seg_id * seg_len
        seg_e = min((seg_id + 1) * seg_len, test_time_step)
        lrec, lobs, lact = test_epoch(rank, use_gpu, world_size, dataloader, seg_b, seg_e, model, main, device, 0)
        lobs=torch.mean(lobs, dim=0).cpu().tolist()
        lact=torch.mean(lact, dim=0).cpu().tolist()
        lobss.extend(lobs)
        lacts.extend(lact)
        lrecs.append(lrec)

    if(main):
        print("\n\n[Results]")
        print("\n\n[Reconstruct Loss]", numpy.mean(lrecs))
        print("\n\n[Prediction Loss]")
        print("\n".join(map(str, lobss)))
        print("\n\n[Action Cross Entropy]")
        print("\n".join(map(str, lacts)))

def test_epoch(rank, use_gpu, world_size, dataloader, start, end, model, main, device, epoch_id):
    # Example training loop
    results = []
    all_length = len(dataloader)

    if(main):
        print("[EVALUATION] Epochs: %s..." % epoch_id)

    lobss = []
    lacts = []
    lrecs = 0.0
    cnts = 0
    for batch_idx, batch in enumerate(dataloader):
        obs,behavior_acts,label_acts,rews = batch
        obs = obs.to(device)[:, start:(end+1)]
        behavior_acts = behavior_acts.to(device)[:, start:end]
        label_acts = label_acts.to(device)[:, start:end]
        obs = obs.permute(0, 1, 4, 2, 3)
        length = label_acts.shape[0] * label_acts.shape[1]
        with torch.no_grad():
            lrec = model.module.vae_loss(obs)
            lobs, _, lact, cnt = model.module.sequential_loss(obs, behavior_acts, label_acts, reduce=None)

        lrecs += lrec.cpu() * cnt
        cnts += cnt
        lobss.append(lobs)
        lacts.append(lact)

        if(main):
            show_bar((batch_idx + 1) / all_length, 100)
            sys.stdout.flush()

    lrecs/=cnts
    lrecs = lrecs.item()
    lobss=torch.stack(lobss, dim=0)
    lacts=torch.stack(lacts, dim=0)

    return lrecs, lobss, lacts

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
             args=(use_gpu, world_size, config, 0),
             nprocs=world_size if use_gpu else min(world_size, 4),  # Limit CPU processes if desired
             join=True)
