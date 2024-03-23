import os
import sys
import argparse
import torch
import numpy
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from reader import MazeDataSet
from models import MazeModels
from torch.cuda.amp import autocast

os.environ['MASTER_ADDR'] = 'localhost'  # Example IP address, replace with your master node's IP
os.environ['MASTER_PORT'] = '12345'        # Example port, choose an available port

def show_bar(fraction, bar):
    percentage = int(bar * fraction)
    empty = bar - percentage
    sys.stdout.write("\r") 
    sys.stdout.write("[") 
    sys.stdout.write("=" * percentage)
    sys.stdout.write(" " * empty)
    sys.stdout.write("]") 
    sys.stdout.write("%.2f %%" % (percentage * 100 / bar))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_path(save_model_path, epoch_id):
    directory_path = '%s/%02d/' % (save_model_path, epoch_id)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return ('%s/model.pth' % directory_path,'%s/optimizer.pth' % directory_path) 

def main_epoch(rank, use_gpu, world_size, max_epochs, batch_size,
        train_data_path, test_data_path, 
        load_model_path, save_model_path, 
        time_step, learning_rate, main_rank):
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
    model = MazeModels(image_size=128, map_size=7, action_size=5, max_steps=time_step)
    if(main):
        print("Number of parameters: ", count_parameters(model))
        print("Number of parameters in Encoder: ", count_parameters(model.encoder))
        #print("Number of parameters in Temporal Encoder: ", count_parameters(model.temporal_encoder_1) * 3)
        print("Number of parameters in Observation Decoder: ", count_parameters(model.decoder))
        print("Number of parameters in Action Decoder: ", count_parameters(model.action_decoder))
        print("Number of parameters in Map Decoder: ", count_parameters(model.map_decoder))


    model = model.to(device)

    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    # Example dataset and dataloader
    if(main):
        print("Initializing Training Dataset...")
    train_dataset = MazeDataSet(train_data_path, time_step, verbose=main)
    if(main):
        print("Initializing Testing Dataset...")
    test_dataset = MazeDataSet(test_data_path, time_step, verbose=main)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1.0e-5)

    if(load_model_path is not None):
        model.load_state_dict(torch.load('%s/model.pth' % load_model_path))
        optimizer.load_state_dict(torch.load('%s/optimizer.pth' % load_model_path))

    test_epoch(rank, use_gpu, world_size, test_dataloader, model, main, device, 0)
    # Example training loop
    total_iteration = len(train_dataloader)
    epoch_repeat = 1
    for epoch_id in range(max_epochs):
        for epoch_repeat_id in range(epoch_repeat):
            for batch_idx, batch in enumerate(train_dataloader):
                obs, acts, rews, maps = batch
                obs = obs.to(device)
                acts = acts.to(device)
                rews = rews.to(device)
                maps = maps.to(device)
                obs = obs.permute(0, 1, 4, 2, 3)
                maps = maps.permute(0, 1, 4, 2, 3)
                with autocast():
                    lrec, lobs, lact, lmap, lrew, cnt = model.module.train_loss(obs, acts, rews, maps)
                loss = 2.0 * lrec + lobs + lact + 0.2 * lmap + lrew
                prt_loss = (lrec.detach().cpu().numpy(), lobs.detach().cpu().numpy(), lact.detach().cpu().numpy(), lmap.detach().cpu().numpy(), lrew.detach().cpu().numpy())

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.module.parameters(), 1.0)

                #for name, param in model.module.named_parameters():
                #    if(param.grad is None):
                #        prt_grad = None
                #    else:
                #        prt_grad = torch.norm(param.grad, torch.inf)
                #    print(f"Gradient of {name}: {prt_grad}")

                optimizer.step()
                scheduler.step()
                if(main):
                    percentage = (batch_idx + 1) / total_iteration * 100
                    print("Epoch: %s [ %.2f %% ] Iteration: %s LearningRate:%f Loss: Recover: %s, Future Prediction MSE: %s, Action Cross Entropy: %s, Map Prediction MSE: %s, Reward Prediction: %s" % 
                            ((epoch_id + 1, percentage, batch_idx, scheduler.get_last_lr()[0]) + prt_loss))
                sys.stdout.flush()

            if(main):
                mod_path, opt_path = model_path(save_model_path, epoch_id)
                torch.save(model.state_dict(), mod_path)
                torch.save(optimizer.state_dict(), opt_path)

        test_epoch(rank, use_gpu, world_size, test_dataloader, model, main, device, epoch_id + 1)

def test_epoch(rank, use_gpu, world_size, test_dataloader, model, main, device, epoch_id):
    # Example training loop
    results = []
    all_length = len(test_dataloader)

    if(main):
        print("[EVALUATION] Epochs: %s..." % epoch_id)
    for batch_idx, batch in enumerate(test_dataloader):
        obs,acts,rews,maps = batch
        obs = obs.to(device)
        acts = acts.to(device)
        rews = rews.to(device)
        maps = maps.to(device)
        obs = obs.permute(0, 1, 4, 2, 3)
        maps = maps.permute(0, 1, 4, 2, 3)
        length = rews.shape[0] * rews.shape[1]
        with torch.no_grad():
            lrec, lobs, lact, lmap, lrew, cnt = model.module.train_loss(obs, acts, rews, maps)
            lrec = cnt * lrec
            lobs = cnt * lobs
            lact = cnt * lact
            lmap = cnt * lmap
            lrew = cnt * lrew

        dist.all_reduce(lrec.data)
        dist.all_reduce(lobs.data)
        dist.all_reduce(lact.data)
        dist.all_reduce(lmap.data)
        dist.all_reduce(lrew.data)
        dist.all_reduce(cnt.data)

        results.append((lrec.cpu(), lobs.cpu(), lact.cpu(), lmap.cpu(), lrew.cpu(), cnt.cpu()))

        if(main):
            show_bar((batch_idx + 1) / all_length, 100)
            sys.stdout.flush()

    sum_lrec = 0
    sum_lobs = 0
    sum_lact = 0
    sum_lrew = 0
    sum_lmap = 0
    sum_cnt = 0
    for lrec, lobs, lact, lmap, lrew, cnt in results:
        sum_lrec += lrec
        sum_lobs += lobs
        sum_lact += lact
        sum_lmap += lmap
        sum_lrew += lrew
        sum_cnt += cnt
    sum_lobs /= sum_cnt
    sum_lrec /= sum_cnt
    sum_lact /= sum_cnt
    sum_lmap /= sum_cnt
    sum_lrew /= sum_cnt

    if(main):
        print("\n[EVALUATION] Epochs: %s; Loss: Recover: %s, Future Prediction MSE: %s, Map Prediction MSE: %s, Action Cross Entropy: %s, Reward Prediction: %s" % 
                (epoch_id, sum_lrec, sum_lobs, sum_lmap, sum_lact, sum_lrew))
        sys.stdout.flush()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--time_step', type=int, default=64)
    parser.add_argument('--save_path', type=str, default='./model/')
    parser.add_argument('--load_path', type=str, default=None)
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_gpu else os.cpu_count()
    if(use_gpu):
        print("Use Parallel GPUs: %s" % world_size)
    else:
        print("Use Parallel CPUs: %s" % world_size)

    mp.spawn(main_epoch,
             args=(use_gpu, world_size, args.max_epochs, args.batch_size,
                    args.train_data_path, args.test_data_path,
                    args.load_path, args.save_path,
                    args.time_step, args.lr, 0),
             nprocs=world_size if use_gpu else min(world_size, 4),  # Limit CPU processes if desired
             join=True)
