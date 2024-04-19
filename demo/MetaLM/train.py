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
os.environ['MASTER_PORT'] = '12343'        # Example port, choose an available port


def main_epoch(rank, use_gpu, world_size, max_epochs, eval_interval, 
        batch_size, vocab_size, train_data_path, test_data_path, 
        load_model_path, save_model_path, 
        max_time_step, train_time_step, learning_rate, main_rank):
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
    model = LMBase(vocab_size=vocab_size,
                hidden_size=1024,
                nhead=16,
                max_time_step=max_time_step,
                n_trn_block=12)
    if(main):
        print("Model parameters:", count_parameters(model))

    model = model.to(device)

    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    # Example dataset and dataloader
    if(main):
        print("Initializing Training Dataset...")
    train_dataset = LMDataSet(train_data_path, 500, verbose=main)
    if(main):
        print("Initializing Testing Dataset...")
    test_dataset = LMDataSet(test_data_path, 500, verbose=main)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda x:noam_scheduler(x, 1000))

    if(load_model_path is not None):
        model = custom_load_model(model, f'{load_model_path}/model.pth')

    # Perform the first evaluation
    test_epoch(rank, use_gpu, world_size, test_dataloader, model, main, device, 0, max_time_step)

    def main_round(rid, dataloader):
        total_iteration = len(dataloader)
        for batch_idx, (feature, label) in enumerate(dataloader):
            feature = feature[:, :train_time_step].to(device)
            label = label[:, :train_time_step].to(device)
            #with autocast():
            loss = model.module.perplexity(feature, label)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.module.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            if(main):
                percentage = (batch_idx + 1) / total_iteration * 100
                print("Epoch: %s [ %.2f %% ][MAIN ROUND] Iteration: %s; Learning_Rate:%s; Perplexity: %s;" % 
                        (rid, percentage, batch_idx, scheduler.get_last_lr()[0], float(loss.detach().cpu().numpy())))
            sys.stdout.flush()

    # Example training loop
    for epoch_id in range(1, max_epochs + 1):
        model.train()
        main_round(epoch_id, train_dataloader)
        if(main and epoch_id % eval_interval == 0):
            mod_path, opt_path_vae, opt_path_seq = model_path(save_model_path, epoch_id)
            torch.save(model.state_dict(), mod_path)
            torch.save(optimizer.state_dict(), opt_path_seq)

        model.eval()
        # Perform the evaluation according to interval
        if(epoch_id % eval_interval == 0):
            test_epoch(rank, use_gpu, world_size, test_dataloader, model, main, device, epoch_id, max_time_step)

def test_epoch(rank, use_gpu, world_size, test_dataloader, model, main, device, epoch_id, max_time_step):
    # Example training loop
    results = []
    all_length = len(test_dataloader)

    if(main):
        print("[EVALUATION] Epochs: %s..." % epoch_id)

    for batch_idx, (feature, label) in enumerate(test_dataloader):
        feature = feature[:, :max_time_step].to(device)
        label = label[:, :max_time_step].to(device)
        #with autocast():
        with torch.no_grad():
            loss = model.module.perplexity(feature, label)
            length = torch.tensor(feature.shape[0] * feature.shape[1]).to(loss.device)

            loss = loss * length

        dist.all_reduce(loss.data)
        dist.all_reduce(length.data)

        results.append((loss.cpu(), length.cpu()))

        if(main):
            show_bar((batch_idx + 1) / all_length, 100)
            sys.stdout.flush()

    sum_loss = 0
    sum_cnt = 0
    for loss, cnt in results:
        sum_loss += loss
        sum_cnt += cnt
    sum_cnt = max(1, sum_cnt)
    sum_loss /= sum_cnt

    if(main):
        print("\n[EVALUATION] Epochs: %s; Perplexity: %s\n"
                % (epoch_id, sum_loss))
        sys.stdout.flush()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--max_time_step', type=int, default=1024)
    parser.add_argument('--train_time_step', type=int, default=256)
    parser.add_argument('--vocab_size', type=int, default=64)
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
             args=(use_gpu, world_size, args.max_epochs, args.eval_interval, 
                    args.batch_size, args.vocab_size,
                    args.train_data_path, args.test_data_path,
                    args.load_path, args.save_path,
                    args.max_time_step, args.train_time_step, args.lr, 0),
             nprocs=world_size if use_gpu else min(world_size, 4),  # Limit CPU processes if desired
             join=True)
