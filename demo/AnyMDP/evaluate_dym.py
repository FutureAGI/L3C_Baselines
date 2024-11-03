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

package_path = '/home/shaopt/code/foundation_model'
sys.path.append(package_path)
from l3c_baselines.dataloader import AnyMDPDataSet, PrefetchDataLoader, segment_iterator
from l3c_baselines.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from l3c_baselines.utils import custom_load_model, noam_scheduler, LinearScheduler
from l3c_baselines.utils import count_parameters, check_model_validity, model_path
from l3c_baselines.utils import Configure, gradient_failsafe, DistStatistics, rewards2go
from l3c_baselines.models import AnyMDPRSA

import gymnasium as gym
from gym.envs.toy_text.frozen_lake import generate_random_map

os.environ['MASTER_ADDR'] = 'localhost'  # Example IP address, replace with your master node's IP

def create_env(env_name):
    if(env_name.lower() == "lake"):
        env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), is_slippery=True)
        return env
    # elif(env_name.lower() == "cliff"):
    #     env = gym.make('CliffWalking-v0')
    #     return env
    # elif(env_name.lower() == "taxi"):
    #     env = gym.make('Taxi-v3')
    #     return env
    # elif(env_name.lower() == "blackjack"):
    #     env = gym.make('Blackjack-v1', natural=False, sab=True)
    #     return env
    else:
        raise ValueError("Unknown env name: {}".format(env_name))
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

def string_mean_var(downsample_length, mean, var):
    string=""
    for i in range(mean.shape[0]):
        string += f'{downsample_length * i}\t{mean[i]}\t{var[i]}\n'
    return string
# config = demo_config
def anymdp_model_epoch(rank, config, env, model, main, device, downsample_length = 10):
    # Example training loop

    state_init, _ = env.reset()
    obs_arr = [state_init]
    act_arr = []
    rew_arr = []
    obs_pred_arr = []
    rew_pred_arr = []
    
    total_step = 1
    success_count = 0
    obs_loss_total = 0.0
    rew_loss_total = 0.0
    # task-wise loss
    obs_loss_list = []
    rew_loss_list = []
    success_rate_list = []
    downsample_scuccess_count = 0

    cache = None
    model.init_mem()
    temperature = config.T_ini
    new_tasks = False
    segment_id = 0
    for task_index in range(config.task_num):
        done = False
        step = 1
        task_start_position = len(rew_arr)
        segment_id = len(rew_arr)//config.seg_len
        segment_start_position = segment_id * config.seg_len
        while not done:
            temperature = max(temperature * (1.0 - config.T_dec), config.T_min)
            state_out, action_out, reward_out, new_cache = model.module.inference_step_by_step(
                observations = obs_arr[segment_start_position:],
                rewards = rew_arr[segment_start_position:],
                behavior_actions = act_arr[segment_start_position:],
                temp = temperature,
                new_tasks = new_tasks,
                device = device,
                cache = cache,
                need_cache = True,
                update_memory = True
            )
            if new_tasks:
                # Remove the init state in next task form the obs_arr
                # If we don't want to remove init state, we need to append 0 value to both act_arr and rew_arr.
                obs_arr.pop(-1) 
            new_tasks = False
            cache = new_cache
            act_arr.append(action_out)
            new_state, new_reward, done, _ = env.step(action_out)
            obs_arr.append(new_state)   
            rew_arr.append(new_reward)
            obs_pred_arr.append(state_out)
            rew_pred_arr.append(reward_out)
            if done:
                new_tasks = True
                if new_reward==1:
                    success_count += 1
                    downsample_scuccess_count += 1
            else:
                step += 1
        # Reset the environment, prepare new task.
        next_task_state_init, _ = env.reset()
        obs_arr.append(next_task_state_init)
        # Start statistics
        # -World model loss
        # --Easy case, obs value is discrete value, if obs is img or continous value, refer to MazeWorld/evaluate.py.
        obs_loss = numpy.mean(obs_arr[task_start_position + 1:] - obs_pred_arr)
        rew_loss = numpy.mean(rew_arr[task_start_position:] - rew_pred_arr)
        # --obs_loss and rew_loss is task-wise loss, for step-wise/position-wise loss, then we don't need to average arcoss step.
        obs_loss_list.append(obs_loss)
        rew_loss_list.append(rew_loss)
        obs_loss_total = (obs_loss_total*total_step + obs_loss*step)/(total_step+step)
        rew_loss_total = (rew_loss_total*total_step + rew_loss*step)/(total_step+step)
        total_step += step
        if task_index % downsample_length == 0:
            success_rate_list.append(downsample_scuccess_count/downsample_length)
            downsample_scuccess_count = 0
    total_success_rate = success_count/config.task_num
    # Todo: Logger
    # 拼接统计数据
    # 拼接并保存obs_arr，act_arr，rew_arr，可根据downsample区间内的成功率来切分不同的数据集，teacher & student

    

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

    train_config = config.train_config
    demo_config = config.demo_config
    
    # Load Model
    model = AnyMDPRSA(config.model_config, verbose=main)
    model = model.to(device)
    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)
    if(demo_config.model_config.has_attr("load_model_path") and 
            demo_config.model_config.load_model_path is not None and 
            demo_config.model_config.load_model_path.lower() != 'none'):
        model = custom_load_model(model, f'{demo_config.model_config.load_model_path}/model.pth', 
                                  black_list=train_config.load_model_parameter_blacklist, 
                                  strict_check=False)
        print("------------Load model success!------------")
    env = create_env(demo_config.env_config.name)
    # Perform the first evaluation
    anymdp_model_epoch(rank, demo_config, env, model, main, device, demo_config.downsample_size)

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

    mp.spawn(anymdp_main_epoch,
             args=(use_gpu, world_size, config, 0, config.run_name),
             nprocs=world_size if use_gpu else min(world_size, 4),  # Limit CPU processes if desired
             join=True)