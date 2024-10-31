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
from l3c_baselines.utils import Logger, log_progress, log_debug, log_warn
from l3c_baselines.utils import custom_load_model, noam_scheduler, LinearScheduler
from l3c_baselines.utils import count_parameters, check_model_validity, model_path
from l3c_baselines.utils import Configure, gradient_failsafe, DistStatistics
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
    log_config = config.log_config

    # Initialize the Dataset and DataLoader
    vae_dataset = MazeDataSet(train_config.data_path, train_config.seq_len_vae, verbose=main)
    causal_dataset = MazeDataSet(train_config.data_path, train_config.seq_len_causal, verbose=main)

    vae_dataloader = PrefetchDataLoader(vae_dataset, batch_size=train_config.batch_size_vae, rank=rank, world_size=world_size)
    causal_dataloader = PrefetchDataLoader(causal_dataset, batch_size=train_config.batch_size_causal, rank=rank, world_size=world_size)
    run_name = config.run_name

    # Initialize the Logger
    logger_causal = Logger("learning_rate", 
                        "loss_worldmodel_raw", 
                        "loss_worldmodel_latent", 
                        "loss_policymodel", 
                        on=main,
                        max_iter=len(causal_dataloader), 
                        use_tensorboard=log_config.use_tensorboard, 
                        log_file=log_config.training_log_causal,
                        prefix=f"{run_name}-Training-Causal",
                        field=f"{log_config.tensorboard_log}/train-{run_name}")


    logger_vae = Logger("sigma", 
                        "lambda",
                        "learning_rate",
                        "reconstruction",
                        "kl-divergence",
                        on=main,
                        max_iter=len(vae_dataloader), 
                        use_tensorboard=log_config.use_tensorboard, 
                        log_file=log_config.training_log_vae,
                        prefix=f"{run_name}-Training-VAE",
                        field=f"{log_config.tensorboard_log}/train-{run_name}")


    logger_eval = []
    for i in range((test_config.seq_len - 1) // test_config.seg_len + 1):
        logger_eval.append(
                Logger("validation_reconstruction", 
                       "validation_worldmodel_raw", 
                       "validation_worldmodel_latent",
                       "validation_policymodel",
                       use_tensorboard=log_config.use_tensorboard,
                       log_file=log_config.evaluation_log,
                       prefix=f"{run_name}-Evaluation-{i}",
                       on=main,
                       field=f"{log_config.tensorboard_log}/validate-{run_name}-Seg{i}"))

    train_stat_vae = DistStatistics("loss_rec", "loss_kl")
    train_stat_causal = DistStatistics("loss_wm_raw", "loss_wm_latent", "loss_pm")

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
                                  verbose=main, 
                                  strict_check=False)

    # Perform the first evaluation
    test_epoch(rank, use_gpu, world_size, test_config, model, main, device, 0, logger_eval)
    scaler = GradScaler()

    # VAE training loop

    def vae_round(rid, dataloader, max_save_iteartions=-1):
        acc_iter = 0
        dataloader.dataset.reset(rid)
        for batch_idx, batch in enumerate(dataloader):
            acc_iter += 1
            for sub_idx, obs, bacti, lacti, bactd, lactd, rews, bevs in segment_iterator(
                            train_config.seq_len_vae, train_config.seg_len_vae, device, (batch[0], 1), *batch[1:]):
                obs = obs.permute(0, 1, 4, 2, 3)
                vae_optimizer.zero_grad()

                with autocast(dtype=torch.bfloat16, enabled=train_config.use_amp):
                    loss = model.module.vae_loss(obs, _sigma=sigma_scheduler())
                    vae_loss = loss["Reconstruction-Error"] + lambda_scheduler() * loss["KL-Divergence"]

                train_stat_vae.add_with_safety(
                    rank,
                    loss_rec = loss["Reconstruction-Error"],
                    loss_kl = loss["KL-Divergence"])

                if(train_config.use_scaler):
                    scaler.scale(vae_loss).backward()
                    gradient_failsafe(model.module, vae_optimizer, scaler)
                    clip_grad_norm_(model.module.parameters(), 1.0)
                else:
                    vae_loss.backward()

            if(train_config.use_scaler):
                scaler.step(vae_optimizer)
                scaler.update()
            else:
                vae_optimizer.step()

            vae_scheduler.step()
            sigma_scheduler.step()
            lambda_scheduler.step()

            stat_res = train_stat_vae()
            logger_vae(sigma_scheduler(), 
                    lambda_scheduler(), 
                    vae_scheduler.get_last_lr()[0],
                    stat_res["loss_rec"],
                    stat_res["loss_kl"],
                    epoch=rid, 
                    iteration=batch_idx)

            if(main and train_config.has_attr("max_save_iterations") 
                            and acc_iter > train_config.max_save_iterations 
                            and train_config.max_save_iterations > 0):
                acc_iter = 0
                log_debug("Check current validity and save model for safe...")
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
                    train_config.seq_len_causal, train_config.seg_len_causal, device, (batch[0], 1), *batch[1:]):
                obs = obs.permute(0, 1, 4, 2, 3)

                causal_optimizer.zero_grad()
                with autocast(dtype=torch.bfloat16, enabled=train_config.use_amp):
                    # Calculate THE LOSS
                    loss = model.module.sequential_loss(
                        obs, bacti, lacti, bevs, state_dropout=0.20, 
                        start_position=start_position)
                    causal_loss = (train_config.lossweight_worldmodel_latent * loss["wm-latent"]
                            + train_config.lossweight_worldmodel_raw * loss["wm-raw"]
                            + train_config.lossweight_policymodel * loss["pm"]
                            + train_config.lossweight_l2 * loss["causal-l2"])
                start_position += bacti.shape[1]

                train_stat_causal.add_with_safety(
                    rank,
                    loss_wm_raw = loss["wm-raw"],
                    loss_wm_latent = loss["wm-latent"],
                    loss_pm = loss["pm"])

                if(train_config.use_scaler):
                    scaler.scale(causal_loss).backward()
                    gradient_failsafe(model.module, causal_optimizer, scaler)
                else:
                    causal_loss.backward()
                clip_grad_norm_(model.module.parameters(), 1.0)

            if(train_config.use_scaler):
                scaler.step(causal_optimizer)
                scaler.update()
            else:
                causal_optimizer.step()
            causal_scheduler.step()

            if(main):
                stat_res = train_stat_causal()
                logger_causal(causal_scheduler.get_last_lr()[0],
                              stat_res["loss_wm_raw"]["mean"], 
                              stat_res["loss_wm_latent"]["mean"], 
                              stat_res["loss_pm"]["mean"], 
                              epoch=rid, 
                              iteration=batch_idx)

            if(main and train_config.has_attr("max_save_iterations") 
                            and acc_iter > train_config.max_save_iterations 
                            and train_config.max_save_iterations > 0):
                acc_iter = 0
                log_debug("Check current validity and save model for safe...")
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
        if(main and epoch_id % train_config.evaluation_interval == 0):
            log_debug(f"Save Model for Epoch-{epoch_id}")
            mod_path, opt_path_vae, opt_path_seq = model_path(train_config.save_model_path, epoch_id)
            torch.save(model.state_dict(), mod_path)

        model.eval()
        # Perform the evaluation according to interval
        if(epoch_id % train_config.evaluation_interval == 0):
            test_epoch(rank, use_gpu, world_size, test_config, model, main, device, epoch_id, logger_eval)

def test_epoch(rank, use_gpu, world_size, config, model, main, device, epoch_id, logger):
    # Example training loop

    dataset = MazeDataSet(config.data_path, config.seq_len, verbose=main)
    dataloader = PrefetchDataLoader(dataset, batch_size=config.batch_size, rank=rank, world_size=world_size)

    all_length = len(dataloader)

    stats = []
    for i in range(len(logger)):
        stats.append(DistStatistics("loss_rec", 
                            "loss_wm_raw", 
                            "loss_wm_latent", 
                            "loss_pm", 
                            "count"))

    log_debug(f"Start Evaluation for Epoch-{epoch_id}", on=main)
    log_progress(0, on=main)

    dataset.reset(0)
    for batch_idx, batch in enumerate(dataloader):
        start_position = 0
        model.module.reset()
        for sub_idx, obs, bacti, lacti, bactd, lactd, rews, bevs in segment_iterator(
                    config.seq_len, config.seg_len, device, (batch[0], 1), *batch[1:]):
            obs = obs.permute(0, 1, 4, 2, 3)

            with torch.no_grad():
                loss_vae = model.module.vae_loss(obs)
                loss = model.module.sequential_loss(
                            obs, bacti, lacti, bevs, start_position=start_position)

            stats[sub_idx].add_with_safety(rank, 
                                loss_rec=loss_vae["Reconstruction-Error"], 
                                loss_wm_raw=loss["wm-raw"], 
                                loss_wm_latent=loss["wm-latent"], 
                                loss_pm=loss["pm"],
                                count=loss["count"])

            start_position += bacti.shape[1]

        log_progress((batch_idx + 1) / all_length, on=main)

    for stat, log in zip(stats, logger):
        stat_res = stat()
        log(stat_res["loss_rec"]["mean"], 
                stat_res["loss_wm_raw"]["mean"], 
                stat_res["loss_wm_latent"]["mean"], 
                stat_res["loss_pm"]["mean"], 
                epoch=epoch_id)

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
