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

package_path = '/home/shaopt/code/test/foundation_model'
sys.path.insert(0,package_path)
from l3c_baselines.dataloader import AnyMDPDataSet, PrefetchDataLoader, segment_iterator
from l3c_baselines.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from l3c_baselines.utils import custom_load_model, noam_scheduler, LinearScheduler
from l3c_baselines.utils import count_parameters, check_model_validity, model_path
from l3c_baselines.utils import Configure, DistStatistics, rewards2go
from l3c_baselines.models import AnyMDPRSA

import gymnasium as gym
from gym.envs.toy_text.frozen_lake import generate_random_map

os.environ['MASTER_ADDR'] = 'localhost'  # Example IP address, replace with your master node's IP

def create_env(env_name):
    if(env_name.lower() == "lake"):
        #env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), is_slippery=True)
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, max_episode_steps=1000)
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

    new_state, _ = env.reset()
    obs_arr = [new_state]
    act_arr = []
    rew_arr = []
    obs_pred_arr = []
    rew_pred_arr = []
    
    success_count = 0
    success_rate_list = []
    obs_loss_list = []
    rew_loss_list = []
    downsample_scuccess_count = 0
    state_correct_prob = 0.0
    reward_correct_prob = 0.0
    interviel_step_count = 0

    temperature = config.model_config.policy.T_ini
    state_out_prob_list = None
    env_state = None
    previous_env_state = None
    env_action = None
    for task_index in range(config.task_num):
        done = False
        step = 1
        while not done:
            temperature = max(temperature * (1.0 - config.model_config.policy.T_dec), config.model_config.policy.T_min)
            with torch.no_grad():
              state_out, action_out, reward_out = model.module.generate(
                None,
                new_state,
                temp = temperature,
                device=device)
            reward_out_prob_list = reward_out
            #reward_out = torch.multinomial(reward_out.squeeze(1), num_samples=1).squeeze(1)
            #reward_out = reward_out.squeeze(0).cpu().numpy()
            #reward_out = int(reward_out.item())
            #rew_pred_arr.append(reward_out)
            privous_state = new_state
            
            # debg only
            previous_env_state = new_state
            env_action = action_out
            
            # interact with env
            new_state, new_reward, done, *_ = env.step(action_out)
            
            # Test: change reward, die-> -10， alive-> 1, goal->10
            # if done:
            #     if new_reward == 1:
            #         new_reward = 10
            #     else:
            #         new_reward = -10
            # else:
            #     new_reward = 1

            # collect data
            act_arr.append(action_out)                     
            obs_arr.append(new_state)   
            rew_arr.append(new_reward)
            # world model reward prediction correct count:
            # reward_correct_prob += reward_out_prob_list[0,0, int(new_reward)].item()
            reward_correct_prob += numpy.abs(reward_out_prob_list[0,0,0].item() - new_reward)
            # start learning
            with torch.no_grad():
              state_out, action_out, reward_out = model.module.learn(
                None,
                privous_state,
                action_out,
                new_reward,
                temp = temperature,
                device=device)
            state_out_prob_list = state_out[0,0,:16] / state_out[0,0,:16].sum(dim=-1,keepdim=True)
            #state_out = torch.multinomial(state_out.squeeze(1), num_samples=1).squeeze(1)
            #state_out = state_out.squeeze(0).cpu().numpy()
            #state_out = int(state_out.item())
            #obs_pred_arr.append(state_out)
            # world model state prediction correct count:
            # print("state_out_prob_list = ", state_out_prob_list)
            state_correct_prob += state_out_prob_list[int(new_state)].item()
            
            # debug:
            env_state = int(new_state)
            
            # Trail finish or continue:
            if done: 
                interviel_step_count += step
                if new_reward==1:
                    success_count += 1
                    downsample_scuccess_count += 1
            else:
                step += 1
        # Reset the environment, prepare new task.
        new_state, _ = env.reset()
        obs_arr.append(new_state)
        # Start statistics
        if task_index > 0 and task_index % downsample_length == 0:
            success_rate_list.append(downsample_scuccess_count/downsample_length)
            downsample_scuccess_count = 0
            obs_loss_list.append(state_correct_prob/interviel_step_count)
            rew_loss_list.append(reward_correct_prob/interviel_step_count)
            interviel_step_count = 0
            state_correct_prob = 0.0
            reward_correct_prob = 0.0
            print("trail idx = ", task_index)
            print("contex length = ", len(act_arr))
            print(downsample_length," interval success rate = ", success_rate_list[-1])
            print(downsample_length," interval obs loss = ", obs_loss_list[-1])
            print(downsample_length," interval rew loss = ", rew_loss_list[-1])
            print("previous_env_state = ", previous_env_state)
            print("env_action = ", env_action)
            print("state_out_prob_list = ", state_out_prob_list)
            print("env_state = ", env_state)

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
