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
from torch.cuda.amp import autocast, GradScaler
from dataloader import MazeDataSet, PrefetchDataLoader, segment_iterator
from utils import custom_load_model, noam_scheduler, LinearScheduler
from utils import count_parameters, check_model_validity, model_path
from utils import Configure, gradient_failsafe
from models import MazeModelBase
from collections import defaultdict
from restools.logging import Logger, log_progress, log_debug, log_warn

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

    log_debug("Main gpu", use_gpu, "rank:", rank, device, on=main)

    # Create model and move it to GPU with id `gpu`
    model = MazeModelBase(config.model_config)
    log_debug("Number of parameters: ", count_parameters(model), on=main)
    log_debug("Number of parameters decision transformer: ", count_parameters(model.decformer), on=main)

    model = model.to(device)

    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)


    # Example dataset and dataloader
    train_config = config.train_config
    load_model_path = train_config.load_model_path
    load_model_parameter_blacklist = train_config.load_model_parameter_blacklist
    learning_rate_vae = train_config.learning_rate_vae
    learning_rate_vae_decay_interval = train_config.learning_rate_vae_decay_interval
    learning_rate_causal = train_config.learning_rate_causal
    learning_rate_causal_decay_interval = train_config.learning_rate_causal_decay_interval
    batch_size_vae = train_config.batch_size_vae
    batch_size_causal = train_config.batch_size_causal
    time_step_vae = train_config.time_step_vae
    time_step_causal = train_config.time_step_causal
    data_path = train_config.data_path

    vae_dataset = MazeDataSet(data_path, time_step_vae, verbose=main)
    causal_dataset = MazeDataSet(data_path, time_step_causal, verbose=main)

    vae_dataloader = PrefetchDataLoader(vae_dataset, batch_size=batch_size_vae, rank=rank, world_size=world_size)
    causal_dataloader = PrefetchDataLoader(causal_dataset, batch_size=batch_size_causal, rank=rank, world_size=world_size)

    if(main):
        logger_causal = Logger("iteration", "segment", "learning_rate", "loss_wm", "loss_z", "loss_pm", sum_iter=len(causal_dataloader), use_tensorboard=True)
        logger_vae = Logger("iteration", "segment", "sigma", "lambda", "learning_rate", "loss", sum_iter=len(vae_dataloader))

    sigma_scheduler = train_config.sigma_scheduler
    sigma_value = train_config.sigma_value

    lambda_scheduler = train_config.lambda_scheduler
    lambda_value = train_config.lambda_value

    sigma_scheduler = LinearScheduler(sigma_scheduler, sigma_value)
    lambda_scheduler = LinearScheduler(lambda_scheduler, lambda_value)

    vae_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_vae)
    causal_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_causal)

    lr_scheduler_vae = lambda x:noam_scheduler(x, learning_rate_vae_decay_interval)
    lr_scheduler_causal = lambda x:noam_scheduler(x, learning_rate_causal_decay_interval)

    vae_scheduler = LambdaLR(vae_optimizer, lr_lambda=lr_scheduler_vae)
    causal_scheduler = LambdaLR(causal_optimizer, lr_lambda=lr_scheduler_causal)

    load_model_path = train_config.load_model_path
    save_model_path = train_config.save_model_path
    eval_interval = train_config.evaluate_epochs

    segment_length = train_config.segment_length

    vae_scheduler.step(train_config.vae_start_step)
    causal_scheduler.step(train_config.causal_start_step)

    if(load_model_path is not None):
        model = custom_load_model(model, f'{load_model_path}/model.pth', black_list=load_model_parameter_blacklist, strict_check=False)

    # Perform the first evaluation
    test_config = config.test_config
    test_epoch(rank, use_gpu, world_size, test_config, model, main, device, 0)
    lossweight_worldmodel_latent = train_config.lossweight_worldmodel_latent
    lossweight_worldmodel_raw = train_config.lossweight_worldmodel_raw
    lossweight_policymodel = train_config.lossweight_policymodel
    use_amp = train_config.use_amp
    scaler = GradScaler()

    def vae_round(rid, dataloader, max_save_iteartions=-1):
        acc_iter = 0
        for batch_idx, batch in enumerate(dataloader):
            acc_iter += 1
            for sub_idx, obs, bacts, lacts, rews, targets in segment_iterator(time_step_vae, segment_length, device, *batch):
                obs = obs.permute(0, 1, 4, 2, 3)
                vae_optimizer.zero_grad()

                with autocast(dtype=torch.bfloat16, enabled=use_amp):
                    vae_loss = model.module.vae_loss(obs, _sigma=sigma_scheduler(), _lambda=lambda_scheduler())

                scaler.scale(vae_loss).backward()
                clip_grad_norm_(model.module.parameters(), 1.0)
                scaler.step(vae_optimizer)
                scaler.update()

                vae_scheduler.step()
                sigma_scheduler.step()
                lambda_scheduler.step()

                if(main):
                    lr = vae_scheduler.get_last_lr()[0]
                    lrec = float(vae_loss.detach().cpu().numpy())
                    logger_vae.log(batch_idx, sub_idx, sigma_schedular(), lambda_scheduler(), lr, lrec, epoch=rid, iteration=batch_idx, prefix="VAE")

            if(main and acc_iter > max_save_iterations and max_save_iterations > 0):
                acc_iter = 0
                print("Check current validity and save model for safe...")
                check_model_validity(model.module)
                mod_path, _, _ = model_path(save_model_path, epoch_id)
                torch.save(model.state_dict(), mod_path)

    def causal_round(rid, dataloader, max_save_iterations=-1):
        acc_iter = 0
        for batch_idx, batch in enumerate(dataloader):
            acc_iter += 1
            for sub_idx, obs, bacts, lacts, rews, targets in segment_iterator(time_step_causal, segment_length, device, *batch):
                obs = obs.permute(0, 1, 4, 2, 3)

                causal_optimizer.zero_grad()
                with autocast(dtype=torch.float16, enabled=use_amp):
                    lobs, lz, lact, cnt = model.module.sequential_loss(obs, bacts, lacts, targets, state_dropout=0.20)
                    causal_loss = lossweight_worldmodel_latent * lz + lossweight_worldmodel_raw * lobs + lossweight_policymodel * lact

                scaler.scale(causal_loss).backward()

                gradient_failsafe(model.module, causal_optimizer, scaler)

                clip_grad_norm_(model.module.parameters(), 1.0)
                scaler.step(causal_optimizer)
                scaler.update()

                causal_optimizer.step()
                causal_scheduler.step()

                if(main):
                    lr = causal_scheduler.get_last_lr()[0]
                    fobs = float(lobs.detach().cpu().numpy())
                    fz = float(lz.detach().cpu().numpy())
                    fact = float(lact.detach().cpu().numpy())
                    logger_causal.log(batch_idx, sub_idx, lr, fobs, fz, fact, epoch=rid, iteration=batch_idx, prefix="CAUSAL")

            if(acc_iter > max_save_iterations and max_save_iterations > 0 and main):
                acc_iter = 0
                print("Check current validity and save model for safe...")
                sys.stdout.flush()
                check_model_validity(model.module)
                mod_path, _, _ = model_path(save_model_path, epoch_id)
                torch.save(model.state_dict(), mod_path)

    # Example training loop
    vae_stop_epoch = train_config.epoch_vae_stop
    causal_start_epoch = train_config.epoch_causal_start
    max_save_iterations = train_config.max_save_iterations

    for epoch_id in range(1, train_config.max_epochs + 1):
        model.train()
        if(vae_stop_epoch > epoch_id):
            vae_round(epoch_id, vae_dataloader)
        if(epoch_id > causal_start_epoch):
            # To save the model within an epoch to prevent failure at the end of the epoch
            causal_round(epoch_id, causal_dataloader, max_save_iterations=max_save_iterations)
        if(main):  # Check whether model is still valid
            check_model_validity(model.module)
        if(main and epoch_id % eval_interval == 0):
            mod_path, opt_path_vae, opt_path_seq = model_path(save_model_path, epoch_id)
            torch.save(model.state_dict(), mod_path)

        model.eval()
        # Perform the evaluation according to interval
        if(epoch_id % eval_interval == 0):
            test_epoch(rank, use_gpu, world_size, test_config, model, main, device, epoch_id)

def test_epoch(rank, use_gpu, world_size, config, model, main, device, epoch_id):
    # Example training loop
    load_model_path = config.load_model_path
    batch_size = config.batch_size
    data_path = config.data_path
    time_step = config.time_step
    segment_length = config.segment_length

    dataset = MazeDataSet(data_path, time_step, verbose=main)
    dataloader = PrefetchDataLoader(dataset, batch_size=batch_size, rank=rank, world_size=world_size)

    results = []
    all_length = len(dataloader)

    log_debug("[EVALUATION] Epochs: %s..." % epoch_id, on=main)
    for batch_idx, batch in enumerate(dataloader):
        for sub_idx, obs, bacts, lacts, rews, targets in segment_iterator(time_step, segment_length, device, *batch):
            obs = obs.permute(0, 1, 4, 2, 3)
            length = lacts.shape[0] * lacts.shape[1]

            with torch.no_grad():
                lobs, lat, lact, cnt = model.module.sequential_loss(obs, bacts, lacts, targets)

                lobs = cnt * lobs
                lact = cnt * lact
                lat = cnt * lat

            if torch.isinf(lobs).any() or torch.isnan(lobs).any():
                log_warn(f"Device-{rank} lobs loss = NAN/INF, skip")
                lobs.fill_(0.0)
            if torch.isinf(lact).any() or torch.isnan(lact).any():
                log_warn(f"Device-{rank} lact loss = NAN/INF, skip")
                lact.fill_(0.0)
            if torch.isinf(lat).any() or torch.isnan(lat).any():
                log_warn(f"Device-{rank} lat loss = NAN/INF, skip")
                lat.fill_(0.0)

            dist.all_reduce(lobs.data)
            dist.all_reduce(lact.data)
            dist.all_reduce(lat.data)
            dist.all_reduce(cnt.data)

            results.append((lobs.cpu(), lact.cpu(), lat.cpu(), cnt.cpu()))

            if(main):
                log_progress((batch_idx + 1) / all_length)

    #sum_lrec = 0
    sum_lobs = 0
    sum_lact = 0
    sum_lat = 0
    sum_cnt = 0
    for lobs, lact, lat, cnt in results:
        #sum_lrec += lrec
        sum_lobs += lobs
        sum_lact += lact
        sum_lat += lat
        sum_cnt += cnt
    sum_cnt = max(1, sum_cnt)
    sum_lobs /= sum_cnt
    sum_lact /= sum_cnt
    sum_lat /= sum_cnt

    if(main):
        logger = Logger("loss_wm", "loss_z", "loss_pm")
        logger.log(sum_lobs, sum_lat, sum_lact, epoch=epoch_id)

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
