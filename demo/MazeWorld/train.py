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
from utils import show_bar, count_parameters, model_path
from models import MazeModelBase

os.environ['MASTER_ADDR'] = 'localhost'  # Example IP address, replace with your master node's IP
os.environ['MASTER_PORT'] = '12345'        # Example port, choose an available port


def main_epoch(rank, use_gpu, world_size, max_epochs, eval_interval, 
        vae_stop_epoch, main_start_epoch, batch_size_vae, batch_size_seq,
        train_data_path, test_data_path, 
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
    model = MazeModelBase(image_size=128, map_size=7, action_size=5, max_time_step=max_time_step)
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

    # Example dataset and dataloader
    if(main):
        print("Initializing Training Dataset...")
    train_dataset = MazeDataSet(train_data_path, train_time_step, verbose=main)
    if(main):
        print("Initializing Testing Dataset...")
    test_dataset = MazeDataSet(test_data_path, train_time_step, verbose=main)

    vae_dataloader = PrefetchDataLoader(train_dataset, batch_size=batch_size_vae)
    seq_dataloader = PrefetchDataLoader(train_dataset, batch_size=batch_size_seq)

    test_dataloader = PrefetchDataLoader(test_dataset, batch_size=batch_size_seq)

    sigma_scheduler = LinearScheduler(500, [0, 0, 0.5, 1.0])
    lambda_scheduler = LinearScheduler(500, [0, 1.0e-8, 1.0e-7, 1.0e-6])

    vae_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    main_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    vae_scheduler = LambdaLR(vae_optimizer, lr_lambda=lambda x:noam_scheduler(x, 500, low=1.0e-5))
    main_scheduler = LambdaLR(main_optimizer, lr_lambda=lambda x:noam_scheduler(x, 500, low=1.0e-5))

    if(load_model_path is not None):
        model = custom_load_model(model, f'{load_model_path}/model.pth')

    # Perform the first evaluation
    #test_epoch(rank, use_gpu, world_size, test_dataloader, model, main, device, 0)

    def vae_round(rid, dataloader):
        total_iteration = len(dataloader)
        for batch_idx, batch in enumerate(dataloader):
            obs, acts, rews, maps = batch
            obs = obs.to(device)
            obs = obs.permute(0, 1, 4, 2, 3)
            vae_loss = model.module.vae_loss(obs, _sigma=sigma_scheduler(), _lambda=lambda_scheduler())

            vae_optimizer.zero_grad()
            vae_loss.backward()
            clip_grad_norm_(model.module.parameters(), 1.0)

            vae_optimizer.step()
            vae_scheduler.step()
            sigma_scheduler.step()
            lambda_scheduler.step()
            if(main):
                percentage = (batch_idx + 1) / total_iteration * 100
                print("Epoch: %s [ %.2f %% ][VAE ROUND] Iteration: %s; Hyperparameter: sigma:%f, lambda:%f; LearningRate: %f Reconstruction Loss: %.04f" % 
                        (rid, percentage, batch_idx, sigma_scheduler(), lambda_scheduler(), vae_scheduler.get_last_lr()[0], float(vae_loss.detach().cpu().numpy())))
            sys.stdout.flush()

    def main_round(rid, dataloader):
        total_iteration = len(dataloader)
        for batch_idx, batch in enumerate(dataloader):
            obs, acts, rews, maps = batch
            obs = obs.to(device)
            acts = acts.to(device)
            obs = obs.permute(0, 1, 4, 2, 3)
            lz, lact, cnt = model.module.sequential_loss(obs, acts)
            main_loss = 0.9 * lz + 0.1 * lact

            main_optimizer.zero_grad()
            main_loss.backward()
            clip_grad_norm_(model.module.parameters(), 1.0)

            main_optimizer.step()
            main_scheduler.step()

            if(main):
                percentage = (batch_idx + 1) / total_iteration * 100
                print("Epoch: %s [ %.2f %% ][MAIN ROUND] Iteration: %s; LearningRate:%f; Future Prediction Image: %s;; Action Cross Entropy: %s;" % 
                        (rid, percentage, batch_idx, main_scheduler.get_last_lr()[0],
                            float(lz.detach().cpu().numpy()), float(lact.detach().cpu().numpy()))) 
            sys.stdout.flush()

    # Example training loop
    for epoch_id in range(1, max_epochs + 1):
        model.train()
        if(vae_stop_epoch > epoch_id):
            vae_round(epoch_id, vae_dataloader)
        if(epoch_id > main_start_epoch):
            main_round(epoch_id, seq_dataloader)
        if(main and epoch_id % eval_interval == 0):
            mod_path, opt_path_vae, opt_path_seq = model_path(save_model_path, epoch_id)
            torch.save(model.state_dict(), mod_path)
            torch.save(vae_optimizer.state_dict(), opt_path_vae)
            torch.save(main_optimizer.state_dict(), opt_path_seq)

        model.eval()
        # Perform the evaluation according to interval
        if(epoch_id % eval_interval == 0):
            test_epoch(rank, use_gpu, world_size, test_dataloader, model, main, device, epoch_id)

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
        obs = obs.permute(0, 1, 4, 2, 3)
        length = acts.shape[0] * acts.shape[1]
        with torch.no_grad():
            lrec = model.module.vae_loss(obs)
            lz, lact, cnt = model.module.sequential_loss(obs, acts)

            lrec = cnt * lrec
            lz = cnt * lz
            lact = cnt * lact

        dist.all_reduce(lrec.data)
        dist.all_reduce(lz.data)
        dist.all_reduce(lact.data)
        dist.all_reduce(cnt.data)

        results.append((lrec.cpu(), lz.cpu(), lact.cpu(), cnt.cpu()))

        if(main):
            show_bar((batch_idx + 1) / all_length, 100)
            sys.stdout.flush()

    sum_lrec = 0
    sum_lz = 0
    sum_lact = 0
    sum_cnt = 0
    for lrec, lz, lact, cnt in results:
        if torch.isinf(lrec).any() or torch.isnan(lrec).any():
            print("[WARNING] lrec = NAN")
            continue
        if torch.isinf(lz).any() or torch.isnan(lz).any():
            print("[WARNING] lz = NAN")
            continue
        if torch.isinf(lact).any() or torch.isnan(lact).any():
            print("[WARNING] lact = NAN")
            continue
        if torch.isinf(cnt).any() or torch.isnan(cnt).any():
            print("[Warning]: cnt = NAN")
            continue
        sum_lrec += lrec
        sum_lz += lz
        sum_lact += lact
        sum_cnt += cnt
    sum_cnt = max(1, sum_cnt)
    sum_lrec /= sum_cnt
    sum_lz /= sum_cnt
    sum_lact /= sum_cnt

    if(main):
        print("\n[EVALUATION] Epochs: %s; [Loss] Reconstruction: %s, Future Prediction Image: %s; Action Cross Entropy: %s;" % 
                (epoch_id, sum_lrec, sum_lz, sum_lact))
        sys.stdout.flush()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--vae_batch_size', type=int, default=4)
    parser.add_argument('--sequential_batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--vae_stop_epoch', type=int, default=10)
    parser.add_argument('--main_start_epoch', type=int, default=-1)
    parser.add_argument('--max_time_step', type=int, default=1024)
    parser.add_argument('--train_time_step', type=int, default=256)
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
                    args.vae_stop_epoch, args.main_start_epoch,
                    args.vae_batch_size, args.sequential_batch_size, 
                    args.train_data_path, args.test_data_path,
                    args.load_path, args.save_path,
                    args.max_time_step, args.train_time_step, args.lr, 0),
             nprocs=world_size if use_gpu else min(world_size, 4),  # Limit CPU processes if desired
             join=True)
