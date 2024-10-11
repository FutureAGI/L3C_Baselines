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
from collections import defaultdict

from l3c_baselines.dataloader import MazeDataSet, PrefetchDataLoader, segment_iterator
from l3c_baselines.utils import custom_load_model, noam_scheduler, LinearScheduler
from l3c_baselines.utils import show_bar, count_parameters, check_model_validity, model_path
from l3c_baselines.utils import Configure, Logger, gradient_failsafe, DistStatistics
from l3c_baselines.models import E2EObjNavSA

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
    model = E2EObjNavSA(config.model_config, verbose=main)

    model = model.to(device)

    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    # Example dataset and dataloader
    train_config = config.train_config
    test_config = config.test_config

    # Initialize the Dataset and DataLoader
    vae_dataset = MazeDataSet(train_config.data_path, train_config.seq_len_vae, verbose=main)
    causal_dataset = MazeDataSet(train_config.data_path, train_config.seq_len_causal, verbose=main)

    vae_dataloader = PrefetchDataLoader(vae_dataset, batch_size=train_config.batch_size_vae, rank=rank, world_size=world_size)
    causal_dataloader = PrefetchDataLoader(causal_dataset, batch_size=train_config.batch_size_causal, rank=rank, world_size=world_size)

    # Initialize the Logger
    if(main):
        logger_causal = Logger("iteration", "segment", "learning_rate", "loss_worldmodel_raw", "loss_worldmodel_latent", "loss_policymodel", sum_iter=len(causal_dataloader), use_tensorboard=True)
        logger_vae = Logger("iteration", "segment", "sigma", "lambda", "learning_rate", "reconstruction", "kl-divergence", sum_iter=len(vae_dataloader), 
                use_tensorboard=True)

    # Initialize the loss weight schedulers (for vae)
    sigma_scheduler = train_config.sigma_scheduler
    sigma_value = train_config.sigma_value

    lambda_scheduler = train_config.lambda_scheduler
    lambda_value = train_config.lambda_value

    sigma_scheduler = LinearScheduler(sigma_scheduler, sigma_value)
    lambda_scheduler = LinearScheduler(lambda_scheduler, lambda_value)

    # Initialize the optimizers
    vae_optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr_vae)
    causal_optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr_causal)

    # Initialize the learning rate schedulers
    lr_scheduler_vae = lambda x:noam_scheduler(x, train_config.lr_vae_decay_interval)
    lr_scheduler_causal = lambda x:noam_scheduler(x, train_config.lr_causal_decay_interval)

    vae_scheduler = LambdaLR(vae_optimizer, lr_lambda=lr_scheduler_vae)
    causal_scheduler = LambdaLR(causal_optimizer, lr_lambda=lr_scheduler_causal)

    vae_scheduler.step(train_config.lr_vae_start_step)
    causal_scheduler.step(train_config.lr_causal_start_step)

    # Load the model if specified in the configuration
    if(train_config.has_attr("load_model_path") and 
            train_config.load_model_path is not None and 
            train_config.load_model_path.lower() != 'none'):
        model = custom_load_model(model, f'{train_config.load_model_path}/model.pth', 
                                  black_list=train_config.load_model_parameter_blacklist, 
                                  strict_check=False)

    # Perform the first evaluation
    test_epoch(rank, use_gpu, world_size, test_config, model, main, device, 0)
    scaler = GradScaler()

    # VAE training loop

    def vae_round(rid, dataloader, max_save_iteartions=-1):
        acc_iter = 0
        dataloader.dataset.reset(rid)
        for batch_idx, batch in enumerate(dataloader):
            acc_iter += 1
            for sub_idx, obs, bacti, lacti, bactd, lactd, rews, bevs in segment_iterator(
                            train_config.seq_len_vae, train_config.seg_len_vae, device, *batch):
                obs = obs.permute(0, 1, 4, 2, 3)
                vae_optimizer.zero_grad()

                with autocast(dtype=torch.bfloat16, enabled=train_config.use_amp):
                    loss = model.module.vae_loss(obs, _sigma=sigma_scheduler())
                    vae_loss = loss["Reconstruction-Error"] + lambda_scheduler() * loss["KL-Divergence"]

                if(use_scaler):
                    scaler.scale(vae_loss).backward()
                    gradient_failsafe(model.module, vae_optimizer, scaler)
                    clip_grad_norm_(model.module.parameters(), 1.0)
                    scaler.step(vae_optimizer)
                    scaler.update()
                else:
                    vae_loss.backward()
                    vae_optimizer.step()

                vae_scheduler.step()
                sigma_scheduler.step()
                lambda_scheduler.step()

                if(main):
                    lr = vae_scheduler.get_last_lr()[0]
                    lrec = float(loss["Reconstruction-Error"].detach().cpu().numpy())
                    lkl = float(loss["KL-Divergence"].detach().cpu().numpy())
                    logger_vae.log(batch_idx, sub_idx, sigma_scheduler(), lambda_scheduler(), 
                            lr, lrec, lkl, epoch=rid, iteration=batch_idx, prefix="VAE")

            if(main and train_config.has_attr("max_save_iterations") 
                            and acc_iter > train_config.max_save_iterations 
                            and train_config.max_save_iterations > 0):
                acc_iter = 0
                print("Check current validity and save model for safe...")
                check_model_validity(model.module)
                mod_path, _, _ = model_path(train_config.save_model_path, epoch_id)
                torch.save(model.state_dict(), mod_path)

    def causal_round(rid, dataloader, max_save_iterations=-1):
        acc_iter = 0
        dataloader.dataset.reset(rid)
        for batch_idx, batch in enumerate(dataloader):
            acc_iter += 1
            # Important: Must reset the model before each segment
            model.module.reset()
            start_position = 0
            for sub_idx, obs, bacti, lacti, bactd, lactd, rews, bevs in segment_iterator(
                        train_config.seq_len_causal, train_config.seg_len_causal, device, *batch):
                obs = obs.permute(0, 1, 4, 2, 3)

                causal_optimizer.zero_grad()
                with autocast(dtype=torch.bfloat16, enabled=use_amp):
                    # Calculate THE LOSS
                    loss = model.module.sequential_loss(
                        obs, bacti, lacti, bevs, state_dropout=0.20, 
                        start_position=start_position)
                    causal_loss = (train_config.lossweight_worldmodel_latent * loss["wm-latent"]
                            + train_config.lossweight_worldmodel_raw * loss["wm-raw"]
                            + train_config.lossweight_policymodel * loss["pm"]
                            + train_config.lossweight_l2 * loss["causal-l2"])
                start_position += bacti.shape[1]

                if(train_config.use_scaler):
                    scaler.scale(causal_loss).backward()
                    gradient_failsafe(model.module, causal_optimizer, scaler)
                    clip_grad_norm_(model.module.parameters(), 1.0)
                    scaler.step(causal_optimizer)
                    scaler.update()
                else:
                    causal_loss.backward()
                    causal_optimizer.step()

                causal_optimizer.step()
                causal_scheduler.step()

                if(main):
                    lr = causal_scheduler.get_last_lr()[0]
                    fobs = float(lobs.detach().cpu().numpy())
                    fz = float(lz.detach().cpu().numpy())
                    fact = float(lact.detach().cpu().numpy())
                    logger_causal.log(batch_idx, sub_idx, lr, fobs, fz, fact, epoch=rid, iteration=batch_idx, prefix="CAUSAL")

            if(main and train_config.has_attr("max_save_iterations") 
                            and acc_iter > train_config.max_save_iterations 
                            and train_config.max_save_iterations > 0):                acc_iter = 0
                print("Check current validity and save model for safe...")
                sys.stdout.flush()
                check_model_validity(model.module)
                mod_path, _, _ = model_path(train_config.save_model_path, epoch_id)
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
        if(main and epoch_id % train_config.eval_interval == 0):
            print(f"Save Model for Epoch-{epoch_id}")
                sys.stdout.flush()
            mod_path, opt_path_vae, opt_path_seq = model_path(train_config.save_model_path, epoch_id)
            torch.save(model.state_dict(), mod_path)

        model.eval()
        # Perform the evaluation according to interval
        if(epoch_id % train_config.eval_interval == 0):
            test_epoch(rank, use_gpu, world_size, test_config, model, main, device, epoch_id)

def test_epoch(rank, use_gpu, world_size, config, model, main, device, epoch_id):
    # Example training loop

    dataset = MazeDataSet(config.data_path, config.seq_len, verbose=main)
    dataloader = PrefetchDataLoader(dataset, batch_size=batch_size, rank=rank, world_size=world_size)

    results = []
    all_length = len(dataloader)

    stat = DistStatistics("loss_reconstruction", "loss_wm_raw", "loss_wm_latent", "loss_pm", "count")

    if(main):
        print("[EVALUATION] Epochs: %s..." % epoch_id)
    dataset.reset(0)
    for batch_idx, batch in enumerate(dataloader):
        start_step = 0
        model.module.reset()
        for sub_idx, obs, bacti, lacti, bactd, lactd, rews, bevs in segment_iterator(
                    config.seq_len, config.seg_len, device, *batch):
            obs = obs.permute(0, 1, 4, 2, 3)

            with torch.no_grad():
                loss_vae = model.module.vae_loss(obs)
                loss = model.module.sequential_loss(
                            obs, bacti, lacti, bevs, start_step=start_step)

                lrec = loss_vae["Reconstruction-Error"] * loss["count"]
                lraw = loss["wm-raw"] * loss["count"]
                llat = loss["wm-latent"] * loss["count"]
                lpm = loss["pm"] * loss["count"]
                cnt = loss["count"]

            stat.add_with_safty(rank, 
                                loss_reconstruction=lrec, 
                                loss_wm_raw=lraw, 
                                loss_wm_latent=llat, 
                                loss_pm=lpm,
                                count=cnt)

            start_step += bacti.shape[1]

            if(main):
                show_bar((batch_idx + 1) / all_length)
                sys.stdout.flush()

    if(main):
        stat_res = stat()
        logger = Logger(*stat_res.keys())
        logger.log(*stat_res.values(), epoch=epoch_id)

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
