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
from l3c_baselines.utils import count_parameters, check_model_validity, model_path
from l3c_baselines.utils import Configure, gradient_failsafe, DistStatistics, DistStatistics2, rewards2go
from l3c_baselines.models import AnyMDPRSA

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
    mean_loss = torch.mean(loss_matrix, dim=0)
    var_loss = torch.var(loss_matrix, dim=0)

    # Create a new matrix
    result_matrix = torch.cat((mean_loss, var_loss), dim=0)

    return result_matrix


def anymdp_model_epoch(rank, world_size, config, model, main, device, epoch_id, logger, logger_position_wise):
    # Example training loop

    dataset = AnyMDPDataSet(config.data_path, config.seq_len, verbose=main)
    dataloader = PrefetchDataLoader(dataset, batch_size=1, rank=rank, world_size=world_size)

    results = []
    all_length = len(dataloader)

    seg_num = (config.seq_len - 1) // config.seg_len + 1

    stat = [DistStatistics("loss_wm_s", "loss_wm_r", "loss_pm", "count") for _ in range(seg_num)]
    stat2 = [DistStatistics2("loss_wm_s_ds", "loss_wm_r_ds", "loss_pm_ds", "count")]
    if(main):
        log_debug("Start evaluation ...")
        log_progress(0)
    dataset.reset(0)

    downsample_length =  100

    for batch_idx, batch in enumerate(dataloader):
        start_position = 0
        model.module.reset()
        sarr, baarr, laarr, rarr = batch
        r2goarr = rewards2go(rarr)
        loss_wm_s_T = None
        loss_wm_r_T = None
        loss_pm_T = None
        loss_count = None
        for sub_idx, states, bactions, lactions, rewards, r2go in segment_iterator(
                    config.seq_len, config.seg_len, device, 
                    (sarr, 1), baarr, laarr, (rarr, 1), (r2goarr, 1)):
            with torch.no_grad():
                # loss dim is Bxt, t = T // seg_len 
                loss = model.module.sequential_loss(
                            states, bactions, lactions, r2go[:, :-1], r2go[:, 1:], start_position=start_position, reduce_dim=None)

            stat[sub_idx].add_with_safety(rank, 
                                loss_wm_s=loss["wm-s"], 
                                loss_wm_r=loss["wm-r"], 
                                loss_pm=loss["pm"],
                                count=loss["count"])
            if (sub_idx == 0):
                loss_wm_s_T=loss["wm-s"]
                loss_wm_r_T=loss["wm-r"]
                loss_pm_T=loss["pm"]
            else:
                loss_wm_s_T = torch.cat((loss_wm_s_T, loss["wm-s"]), dim=1)
                loss_wm_r_T = torch.cat((loss_wm_r_T, loss["wm-r"]), dim=1)
                loss_pm_T = torch.cat((loss_pm_T, loss["pm"]), dim=1)
            start_position += bactions.shape[1]
        
        # Append over all segment, loss_wm_s_arr, loss_wm_r_arr and loss_pm_arr dim become BxT
        # Downsample over T, dim become [B,T//downsample_length]
        batch_size, seq_length = loss_wm_s_T.shape
        loss_wm_s_ds = torch.mean(loss_wm_s_T.view(batch_size, seq_length//downsample_length, -1), dim=2)
        loss_wm_r_ds = torch.mean(loss_wm_r_T.view(batch_size, seq_length//downsample_length, -1), dim=2)
        loss_pm_ds = torch.mean(loss_pm_T.view(batch_size, seq_length//downsample_length, -1), dim=2)
        # Calculate result matrix, dim become [2,T//downsample_length], first row is position_wise mean, second row is variance.
        stat_loss_wm_s = calculate_result_matrix(loss_wm_s_ds)
        stat_loss_wm_r = calculate_result_matrix(loss_wm_r_ds)
        stat_loss_pm = calculate_result_matrix(loss_pm_ds)
        #Sample length is equal, i.e. 32k, then the count can be batch_size.
        loss_count = batch_size

        stat2.append_with_safety(rank, 
                            loss_wm_s=stat_loss_wm_s, 
                            loss_wm_r=stat_loss_wm_r, 
                            loss_pm=stat_loss_pm,
                            count=loss_count)
        

        if(main):
            log_progress((batch_idx + 1) / all_length)

    

    if(main):
        #log
        for i in range(seg_num):
            stat_res = stat[i]()
            logger[i](stat_res["loss_wm_s"], stat_res["loss_wm_r"], stat_res["loss_pm"],
                    epoch=epoch_id, iteration=epoch_id, prefix=f"EvaluationResults-Seg{i}")
        #log for position-wise loss anylsis
        # dim = [3, T//downsample_length]. Row 1 is mean, row 2 is 90% lower confidence bound, row 3 is 90% upper confidence bound.
        stat_res_wm_s = torch.cat((stat2["loss_wm_s"][0],
                                   (stat2["loss_wm_s"][0] - 1.645*(torch.sqrt(stat2["loss_wm_s"][1]) / torch.sqrt(stat2["count"]))), 
                                   (stat2["loss_wm_s"][0] + 1.645*(torch.sqrt(stat2["loss_wm_s"][1]) / torch.sqrt(stat2["count"])))),
                                   dim = 0)
        stat_res_wm_r = torch.cat((stat2["loss_wm_r"][0],
                                   (stat2["loss_wm_r"][0] - 1.645*(torch.sqrt(stat2["loss_wm_r"][1]) / torch.sqrt(stat2["count"]))),
                                   (stat2["loss_wm_r"][0] + 1.645*(torch.sqrt(stat2["loss_wm_r"][1]) / torch.sqrt(stat2["count"])))),
                                   dim = 0)
        stat_res_pm = torch.cat((stat2["loss_pm"][0],
                                   (stat2["loss_pm"][0] - 1.645*(torch.sqrt(stat2["loss_pm"][1]) / torch.sqrt(stat2["count"]))),
                                   (stat2["loss_pm"][0] + 1.645*(torch.sqrt(stat2["loss_pm"][1]) / torch.sqrt(stat2["count"])))),
                                   dim = 0)
        logger_position_wise(stat_res_wm_s, stat_res_wm_r, stat_res_pm, 
                             epoch=epoch_id, iteration=epoch_id, prefix="EvaluationResults-Positionwise")
        
    for i in range(seg_num):
        stat[i].reset()

def anymdp_main_epoch(rank, use_gpu, world_size, config, main_rank, run_name):
    
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

    demo_config = config.demo_config
    
    # Load Model
    model = AnyMDPRSA(config.model_config, verbose=main)
    model = model.to(device)
    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)
    if(demo_config.has_attr("load_model_path") and 
            demo_config.load_model_path is not None and 
            demo_config.load_model_path.lower() != 'none'):
        model = custom_load_model(model, f'{demo_config.load_model_path}/model.pth', 
                                  black_list=demo_config.load_model_parameter_blacklist, 
                                  strict_check=False)
    
    # Initiate logger
    train_config = config.train_config
    test_config = config.test_config
    if(main):
        eval_seg_num = (test_config.seq_len - 1) // test_config.seg_len + 1
        logger_eval_segment = []
        for i in range(eval_seg_num):
            logger_eval_segment.append(Logger("validation_state_pred", "validation_reward_pred", "validation_policy",
                    sum_iter=train_config.max_epochs, use_tensorboard=True, field=f"runs/validate-{run_name}-Seg{i}"))
        logger_eval_position_wise = []
        logger_eval_position_wise.append(Logger("validation_state_pred_position_wise", "validation_reward_pred_position_wise", "validation_policy_position_wise",
                    sum_iter=train_config.max_epochs, use_tensorboard=True, field=f"runs/validate-{run_name}-PositionWise"))
    else:
        logger_eval = None
        logger_eval_position_wise = None

    # Perform the first evaluation
    anymdp_model_epoch(rank, use_gpu, world_size, test_config, model, main, device, 0, logger_eval, logger_eval_position_wise)

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
    demo_config = config.demo_config
    os.environ['MASTER_PORT'] = demo_config.master_port        # Example port, choose an available port
