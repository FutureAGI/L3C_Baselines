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
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from collections import defaultdict

from l3c_baselines.dataloader import AnyMDPDataSet, PrefetchDataLoader, segment_iterator
from l3c_baselines.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from l3c_baselines.utils import custom_load_model, noam_scheduler, LinearScheduler
from l3c_baselines.utils import count_parameters, check_model_validity, model_path, safety_check, apply_gradient_safely
from l3c_baselines.utils import Configure, DistStatistics, rewards2go
from l3c_baselines.models import AnyMDPRSA

os.environ['MASTER_ADDR'] = 'localhost'  # Example IP address, replace with your master node's IP

def main_epoch(rank, use_gpu, world_size, config, main_rank):
    if use_gpu:
        torch.cuda.set_device(rank)  # Set the current GPU to be used
        device = torch.device(f'cuda:{rank}')
        device_type = 'cuda'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
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
    model = AnyMDPRSA(config.model_config, verbose=main)

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
    dataset = AnyMDPDataSet(train_config.data_path, train_config.seq_len, verbose=main)
    dataloader = PrefetchDataLoader(dataset, batch_size=train_config.batch_size, rank=rank, world_size=world_size)
    run_name = config.run_name

    # Initialize the Logger
    logger = Logger("learning_rate", 
                    "loss_worldmodel_state", 
                    "loss_worldmodel_reward", 
                    "loss_policymodel",
                    "entropy",
                    max_iter=len(dataloader), 
                    use_tensorboard=log_config.use_tensorboard, 
                    log_file=log_config.training_log,
                    prefix=f"{run_name}-Training",
                    on=main,
                    field=f"{log_config.tensorboard_log}/train-{run_name}")

    # Evaluation Logger
    logger_eval = Logger("validation_state_pred", 
                    "validation_reward_pred", 
                    "validation_policy",
                    start_iter=0, 
                    use_tensorboard=log_config.use_tensorboard,
                    log_file=log_config.evaluation_log,
                    prefix=f"{run_name}-Evaluation",
                    on=main,
                    field=f"{log_config.tensorboard_log}/validate-{run_name}")

    train_stat = DistStatistics("loss_worldmodel_state", 
                                "loss_worldmodel_reward", 
                                "loss_policymodel", 
                                "entropy")

    # Initialize the optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    # Initialize the learning rate schedulers
    lr_scheduler = lambda x:noam_scheduler(x, train_config.lr_decay_interval)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_scheduler)

    scheduler.step(train_config.lr_start_step)

    # Load the model if specified in the configuration
    if(train_config.has_attr("load_model_path") and 
            train_config.load_model_path is not None and 
            train_config.load_model_path.lower() != 'none'):
        model = custom_load_model(model, f'{train_config.load_model_path}/model.pth', 
                                  black_list=train_config.load_model_parameter_blacklist, 
                                  strict_check=False, verbose=main)

    # Perform the first evaluation
    #test_epoch(rank, use_gpu, world_size, test_config, model, main, device, 0, logger_eval)

    if(train_config.use_scaler):
        scaler = GradScaler()
    else:
        scaler = None

    # main training loop

    def main_round(rid, dataloader):
        acc_iter = 0
        dataloader.dataset.reset(rid)
        for batch_idx, batch in enumerate(dataloader):
            acc_iter += 1
            # Important: Must reset the model before segment iteration
            model.module.reset()
            start_position = 0
            sarr, baarr, laarr, rarr = batch
            r2goarr = rewards2go(rarr)
            optimizer.zero_grad()
            for sub_idx, states, bactions, lactions, rewards, r2go in segment_iterator(
                        train_config.seq_len, train_config.seg_len, device, 
                        (sarr, 1), baarr, laarr, rarr, (r2goarr, 1)):
                with autocast(dtype=torch.bfloat16, enabled=train_config.use_amp, device_type=device_type):
                    # Calculate THE LOSS
                    loss = model.module.sequential_loss(
                        r2go[:, :-1], # Prompts
                        states, 
                        rewards, # Rewards 
                        bactions, 
                        lactions, 
                        state_dropout=0.20, 
                        reward_dropout=0.20,
                        start_position=start_position)
                    causal_loss = (train_config.lossweight_worldmodel_states * loss["wm-s"]
                            + train_config.lossweight_worldmodel_rewards * loss["wm-r"]
                            + train_config.lossweight_entropy * loss["ent"]
                            + train_config.lossweight_policymodel * loss["pm"]
                            + train_config.lossweight_l2 * loss["causal-l2"])
                start_position += bactions.shape[1]

                train_stat.gather(
                            rank,
                            loss_worldmodel_state = loss["wm-s"],
                            loss_worldmodel_reward = loss["wm-r"],
                            loss_policymodel = loss["pm"],
                            entropy = -loss["ent"],
                            count = loss["count"])

                if(train_config.use_scaler):
                    scaler.scale(causal_loss).backward()
                else:
                    causal_loss.backward()

            apply_gradient_safely(model, optimizer, scaler=scaler)
            scheduler.step()

            # Statistics
            stat_res = train_stat()
            logger(scheduler.get_last_lr()[0], 
                    stat_res["loss_worldmodel_state"]["mean"], 
                    stat_res["loss_worldmodel_reward"]["mean"], 
                    stat_res["loss_policymodel"]["mean"], 
                    stat_res["entropy"]["mean"],
                    epoch=rid,
                    iteration=batch_idx)

            # Safety Check and Save
            if(train_config.has_attr("max_save_iterations") 
                            and acc_iter > train_config.max_save_iterations 
                            and train_config.max_save_iterations > 0):
                acc_iter = 0
                if(main):
                    log_debug("Check current validity and save model for safe...")
                    check_model_validity(model.module)
                    mod_path = model_path(train_config.save_model_path, epoch_id)
                    torch.save(model.state_dict(), mod_path)
                test_epoch(rank, use_gpu, world_size, test_config, model, main, device, epoch_id, logger_eval)

    # Example training loop
    for epoch_id in range(1, train_config.max_epochs + 1):
        model.train()
        # To save the model within an epoch to prevent failure at the end of the epoch
        main_round(epoch_id, dataloader)
        if(main):  # Check whether model is still valid
            check_model_validity(model.module)
        if(main and epoch_id % train_config.evaluation_interval == 0):
            log_debug(f"Save Model for Epoch-{epoch_id}")
            mod_path = model_path(train_config.save_model_path, epoch_id)
            torch.save(model.state_dict(), mod_path)

        # Perform the evaluation according to interval
        if(epoch_id % train_config.evaluation_interval == 0):
            test_epoch(rank, use_gpu, world_size, test_config, model, main, device, epoch_id, logger_eval)

def test_epoch(rank, use_gpu, world_size, config, model, main, device, epoch_id, logger):
    # Example training loop

    dataset = AnyMDPDataSet(config.data_path, config.seq_len, verbose=main)
    dataloader = PrefetchDataLoader(dataset, batch_size=config.batch_size, rank=rank, world_size=world_size)

    results = []
    all_length = len(dataloader)

    seg_num = (config.seq_len - 1) // config.seg_len + 1

    stat = DistStatistics("loss_wm_s", "loss_wm_r", "loss_pm")

    log_debug(f"Start Evaluation for Epoch-{epoch_id}", on=main)
    log_progress(0, on=main)
    dataset.reset(0)
    for batch_idx, batch in enumerate(dataloader):
        start_position = 0
        model.module.reset()
        sarr, baarr, laarr, rarr = batch
        r2goarr = rewards2go(rarr)
        losses = []
        with torch.no_grad():
            for sub_idx, states, bactions, lactions, rewards, r2go in segment_iterator(
                        config.seq_len, config.seg_len, device, 
                        (sarr, 1), baarr, laarr, rarr, (r2goarr, 1)):
                losses.append(model.module.sequential_loss(
                            None,
                            states, 
                            rewards, 
                            bactions, 
                            lactions, 
                            loss_is_weighted=False,
                            start_position=start_position))

            start_position += bactions.shape[1]

        stat.gather(rank, 
                loss_wm_s=[loss["wm-s"] for loss in losses], 
                loss_wm_r=[loss["wm-r"] for loss in losses], 
                loss_pm=[loss["pm"] for loss in losses],
                count=[loss["count"] for loss in losses])

        log_progress((batch_idx + 1) / all_length, on=main)

    stat_res = stat()
    logger(stat_res["loss_wm_s"]["mean"], 
                stat_res["loss_wm_r"]["mean"], 
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
