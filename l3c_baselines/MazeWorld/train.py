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
from torch.utils.data import DataLoader, Dataset
from reader import MazeDataSet
from models import MazeModels, gen_mask
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
        print("Number of parameters in Temporal Encoder: ", count_parameters(model.temporal_encoder_1) * 3)
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=0)

    if(load_model_path is not None):
        model.load_state_dict(torch.load('%s/model.pth' % load_model_path))
        optimizer.load_state_dict(torch.load('%s/optimizer.pth' % load_model_path))

    test_epoch(rank, use_gpu, world_size, test_dataloader, model, main, device, 0)
    # Example training loop
    total_iteration = len(train_dataloader)
    for epoch_id in range(max_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            obs, acts, rews, maps = batch
            obs = obs.to(device)
            acts = acts.to(device)
            rews = rews.to(device)
            maps = maps.to(device)
            obs = obs.permute(0, 1, 4, 2, 3)
            maps = maps.permute(0, 1, 4, 2, 3)
            mask_obs, mask_act = gen_mask(acts, is_train=True)
            mask_obs = mask_obs.to(device)
            mask_act = mask_act.to(device)
            with autocast():
                lmse1, cnt1, lmse2, cnt2, lce, cnt3 = model.module.train_loss(obs, acts, rews, maps, mask_obs, mask_act)
            loss = lmse1 + 0.30 * lmse2 + lce
            prt_loss = (lmse1.detach().cpu().numpy(), lmse2.detach().cpu().numpy(), lce.detach().cpu().numpy())

            try:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                if(main):
                    percentage = (batch_idx + 1) / total_iteration * 100
                    print("Epoch: %s [ %.2f %% ] Iteration: %s LearningRate:%f Loss: Future Prediction MSE: %s, Map Prediction MSE: %s, Action Cross Entropy: %s" % 
                            ((epoch_id + 1, percentage, batch_idx, scheduler.get_last_lr()[0]) + prt_loss))
                sys.stdout.flush()
            except RuntimeError as e:
                print("RuntimeError during backward pass:", e)
                # Additional debugging steps or handling of the error
                break  # Exit the loop or handle the error appropriately

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
        mask_obs, mask_act = gen_mask(acts, is_train=True)
        length = rews.shape[0] * rews.shape[1]
        mask_obs = mask_obs.to(device)
        mask_act = mask_act.to(device)
        with torch.no_grad():
            lmse1, cnt1, lmse2, cnt2, lce, cnt3 = model.module.train_loss(obs, acts, rews, maps, mask_obs, mask_act)
            lmse1 = cnt1 * lmse1
            lmse2 = cnt2 * lmse2
            lce = cnt3 * lce

        dist.all_reduce(lmse1.data)
        dist.all_reduce(lmse2.data)
        dist.all_reduce(lce.data)
        dist.all_reduce(cnt1.data)
        dist.all_reduce(cnt2.data)
        dist.all_reduce(cnt3.data)

        results.append((lmse1.cpu(),lmse2.cpu(),lce.cpu(), cnt1.cpu(), cnt2.cpu(), cnt3.cpu()))

        if(main):
            show_bar((batch_idx + 1) / all_length, 100)
            sys.stdout.flush()

    sum_lmse1 = 0
    sum_lmse2 = 0
    sum_lce = 0
    sum_n1 = 0
    sum_n2 = 0
    sum_n3 = 0
    for lmse1, lmse2, lce, n1, n2, n3 in results:
        sum_lmse1 += lmse1
        sum_lmse2 += lmse2
        sum_lce += lce
        sum_n1 += n1
        sum_n2 += n2
        sum_n3 += n3
    sum_lmse1 /= sum_n1
    sum_lmse2 /= sum_n2
    sum_lce /= sum_n3

    if(main):
        print("\n[EVALUATION] Epochs: %s; Loss: Future Prediction MSE: %s, Map Prediction MSE: %s, Action Cross Entropy: %s" % 
                (epoch_id, sum_lmse1, sum_lmse2, sum_lce))
        sys.stdout.flush()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--lr', type=float, default=5e-4)
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
