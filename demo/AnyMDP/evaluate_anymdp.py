import os
import sys
import argparse
import torch
import numpy
import matplotlib.pyplot as plt
import csv
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

from l3c_baselines.dataloader import AnyMDPDataSet, PrefetchDataLoader, segment_iterator
from l3c_baselines.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from l3c_baselines.utils import custom_load_model, noam_scheduler, LinearScheduler
from l3c_baselines.utils import count_parameters, check_model_validity, model_path, safety_check
from l3c_baselines.utils import Configure, gradient_failsafe, DistStatistics, rewards2go
from l3c_baselines.models import AnyMDPRSA

os.environ['MASTER_ADDR'] = 'localhost'  # Example IP address, replace with your master node's IP
def calculate_result_matrix(loss_matrix):
    """
    Calculate and return a new matrix, where the first row is the result of averaging the input matrix along dim 0,
    and the second row is the result of calculating the variance of the input matrix along dim 0.

    Parameters:
    loss_matrix (torch.Tensor): The input tensor, its shape should be [batch_size, seq_length].

    Returns:
    result_matrix (torch.Tensor): The output tensor, its shape is [2, seq_length].
    """
    # Calculate the mean and variance along dim 0
    mean_loss = []
    var_loss = []
    if loss_matrix.shape[0] > 1:
        mean_loss = torch.mean(loss_matrix, dim=0)
        var_loss = torch.var(loss_matrix, dim=0)
    else:
        mean_loss = loss_matrix
        var_loss = torch.zeros_like(mean_loss)

    # Create a new matrix
    result_matrix = torch.stack((mean_loss, var_loss), dim=0)

    return result_matrix

def string_mean_var(downsample_length, res):
    string=""
    for i, (xm,xb) in enumerate(zip(res["mean"], res["bound"])):
        string += f'{downsample_length * i}\t{xm}\t{xb}\n'
    return string

def anymdp_model_epoch(rank, world_size, config, model_config, model, main, device, downsample_length = 10):
    # Example training loop

    dataset = AnyMDPDataSet(config.data_path, config.seq_len, verbose=main)
    dataloader = PrefetchDataLoader(dataset, batch_size=config.batch_size, rank=rank, world_size=world_size)

    all_length = len(dataloader)

    if config.has_attr("downsample_size") and config.downsample_size is not None:
        downsample_length = config.downsample_size

    stat = DistStatistics("loss_wm_s_ds", "loss_wm_r_ds", "loss_pm_ds")
    log_debug("Start evaluation ...", on=main)
    log_progress(0, on=main)
    dataset.reset(0)

    loss_wm_s_T_batch = None
    loss_wm_r_T_batch = None
    loss_pm_T_batch = None
    loss_count_batch = 0
    for batch_idx, batch in enumerate(dataloader):
        start_position = 0
        model.module.reset()
        sarr, baarr, laarr, rarr = batch
        r2goarr = rewards2go(rarr)
        loss_wm_s_T = []
        loss_wm_r_T = []
        loss_pm_T = []     
        for sub_idx, states, bactions, lactions, rewards, r2go in segment_iterator(
                    config.seq_len, config.seg_len, device, 
                    (sarr, 1), baarr, laarr, rarr, (r2goarr, 1)):
            with torch.no_grad():
                # loss dim is Bxt, t = T // seg_len 
                if model_config.reward_encode.input_type == "Continuous":
                    rewards = rewards.view(rewards.shape[0],rewards.shape[1],1)
                loss = model.module.sequential_loss(
                            None,
                            states, 
                            rewards, 
                            bactions, 
                            lactions, 
                            start_position=start_position,
                            loss_is_weighted=False,
                            reduce_dim=None)
            loss_wm_s_T.append(safety_check(loss["wm-s"]))
            loss_wm_r_T.append(safety_check(loss["wm-r"]))
            loss_pm_T.append(safety_check(loss["pm"]))
            start_position += bactions.shape[1]
        
        # Append over all segment, loss_wm_s_arr, loss_wm_r_arr and loss_pm_arr dim become BxT
        # Downsample over T, dim become [B,T//downsample_length]
        loss_wm_s_T = torch.cat(loss_wm_s_T, dim=1)
        loss_wm_r_T = torch.cat(loss_wm_r_T, dim=1)
        loss_pm_T = torch.cat(loss_pm_T, dim=1)

        bsz, nT = loss_wm_s_T.shape

        nseg = nT // downsample_length
        eff_T = nseg * downsample_length

        loss_wm_s_T = torch.mean(loss_wm_s_T[:, :eff_T].view(bsz, nseg, -1), dim=-1)
        loss_wm_r_T = torch.mean(loss_wm_r_T[:, :eff_T].view(bsz, nseg, -1), dim=-1)
        loss_pm_T = torch.mean(loss_pm_T[:, :eff_T].view(bsz, nseg, -1), dim=-1)
        
        for i in range(bsz):
            stat.gather(rank,
                    loss_wm_s_ds=loss_wm_s_T[i],
                    loss_wm_r_ds=loss_wm_r_T[i],
                    loss_pm_ds=loss_pm_T[i],
                    count=1)
        log_progress((batch_idx + 1) / len(dataloader), on=main)

    stat_res = stat()

    if(main):
        if not os.path.exists(config.output):
            os.makedirs(config.output)
        
        for key_name in stat_res:
            res_text = string_mean_var(downsample_length, stat_res[key_name])
            file_path = f'{config.output}/result_{key_name}.txt'
            if os.path.exists(file_path):
                os.remove(file_path)
            with open(file_path, 'w') as f_model:
                f_model.write(res_text)
        
def anymdp_main_epoch(rank, use_gpu, world_size, config, main_rank):
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

    test_config = config.test_config
    model_config = config.model_config
    
    # Load Model
    model = AnyMDPRSA(config.model_config, verbose=main)
    model = model.to(device)
    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    model = custom_load_model(model, f'{test_config.load_model_path}/model.pth', strict_check=False)
    print("------------Load model success!------------")

    # Perform the first evaluation
    anymdp_model_epoch(rank, world_size, test_config, model_config, model, main, device)

    return


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

    mp.spawn(anymdp_main_epoch,
             args=(use_gpu, world_size, config, 0),
             nprocs=world_size if use_gpu else min(world_size, 4),  # Limit CPU processes if desired
             join=True)
