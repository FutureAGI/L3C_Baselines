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
    model = LMBase(config.model_config)
                
    if(main):
        print("Model parameters:", count_parameters(model))

    model = model.to(device)

    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    # Example dataset and dataloader
    train_config = config.train_config
    load_model_path = train_config.load_model_path
    learning_rate = train_config.learning_rate
    time_step = train_config.time_step
    noam_decay_interval = train_config.learning_rate_noam_decay_interval
    batch_size = config.train_config.batch_size
    max_epochs = train_config.max_epochs

    dataset = LMDataSet(train_config.data_path, train_config.file_size, verbose=main)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda x:noam_scheduler(x, noam_decay_interval))

    if(load_model_path is not None):
        model = custom_load_model(model, f'{load_model_path}/model.pth', strict_check=False)

    # Perform the first evaluation
    test_config = config.test_config
    test_epoch(rank, use_gpu, world_size, test_config, model, main, device, 0)

    def main_round(rid, dataloader):
        total_iteration = len(dataloader)
        for batch_idx, (feature, label) in enumerate(dataloader):
            feature = feature[:, :time_step].to(device)
            label = label[:, :time_step].to(device)
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
        main_round(epoch_id, dataloader)
        if(main and epoch_id % eval_interval == 0):
            mod_path, opt_path_vae, opt_path_seq = model_path(train_config.save_model_path, epoch_id)
            torch.save(model.state_dict(), mod_path)
            torch.save(optimizer.state_dict(), opt_path_seq)

        model.eval()
        # Perform the evaluation according to interval
        if(epoch_id % eval_interval == 0):
            test_epoch(rank, use_gpu, world_size, test_config, model, main, device, epoch_id)

def test_epoch(rank, use_gpu, world_size, config, model, main, device, epoch_id):
    # Example training loop

    data_path = config.data_path
    file_size = config.file_size
    batch_size = config.batch_size
    time_step = config.time_step
    results = []
    dataset = LMDataSet(data_path, file_size, verbose=main)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    all_length = len(dataloader)

    if(main):
        print("[EVALUATION] Epochs: %s..." % epoch_id)

    for batch_idx, (feature, label) in enumerate(dataloader):
        feature = feature[:, :time_step].to(device)
        label = label[:, :time_step].to(device)
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
    os.environ['MASTER_PORT'] = config.train_config.master_port        # Example port, choose an available port

    mp.spawn(main_epoch,
             args=(use_gpu, world_size, config, 0),
             nprocs=world_size if use_gpu else min(world_size, 4),  # Limit CPU processes if desired
             join=True)
