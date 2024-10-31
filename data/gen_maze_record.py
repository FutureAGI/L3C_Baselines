#!/usr/bin/env python
# coding=utf8
# File: dump_maze.py
import gym
import sys
import os
import random
import time
import numpy
import argparse
import multiprocessing
import pickle
import l3c.mazeworld
from l3c.mazeworld import MazeTaskSampler, Resampler
from l3c.mazeworld.agents import SmartSLAMAgent

def run_maze_epoch(
        maze_env,
        max_steps,
        label_memkr=1.0,
        behavior_memkr=0.25,
        behavior_noise=0.20):
    # Must intialize agent after reset
    label_agent = SmartSLAMAgent(maze_env=maze_env, render=False, memory_keep_ratio=label_memkr)
    behavior_agent = SmartSLAMAgent(maze_env=maze_env, render=False, memory_keep_ratio=behavior_memkr)

    done=False
    observation, information = maze_env.reset()
    sum_reward = 0
    reward = 0
    observation_list = [observation]
    cmd_list = [information["command"]]
    bact_id_list = []
    lact_id_list = []
    bact_val_list = []
    lact_val_list = []
    bev_list = []
    reward_list = []
    interval = 0

    behavior_noise_decay = random.random() / max_steps
    step = 0

    while not done:
        lact_id = label_agent.step(observation, reward)
        bact_id = behavior_agent.step(observation, reward)
        true_noise = behavior_noise * (1.0 - behavior_noise_decay * step)
        if(random.random() < true_noise):
            bact_id = maze_env.action_space.sample()

        bact_id_list.append(bact_id)
        lact_id_list.append(lact_id)
        bact_val_list.append(maze_env.list_actions[bact_id])
        lact_val_list.append(maze_env.list_actions[lact_id])

        obs, reward, done, info = maze_env.step(bact_id)
        observation_list.append(observation)
        reward_list.append(reward)
        bev_list.append(maze_env.get_local_map()[1])
        cmd_list.append(info["command"])

        sum_reward += reward
        step += 1

    print("Finish running, sum reward = %f, steps = %d\n"%(sum_reward, len(observation_list)-1))

    return {
            "observations": numpy.array(observation_list, dtype=numpy.uint8),
            "actions_behavior_id": numpy.array(bact_id_list, dtype=numpy.uint8),
            "actions_behavior_val": numpy.array(bact_val_list, dtype=numpy.float32),
            "actions_label_id": numpy.array(lact_id_list, dtype=numpy.uint8),
            "actions_label_val": numpy.array(lact_val_list, dtype=numpy.float32),
            "commands": numpy.array(cmd_list, dtype=numpy.uint8),
            "BEVs": numpy.array(bev_list, dtype=numpy.uint8),
            "rewards": numpy.array(reward_list, dtype=numpy.float32)
            }

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def dump_maze(work_id, path_name, epoch_ids, n_range, 
        label_configs, behavior_configs, 
        max_steps, tasks_from_file):
    # Tasks in Sequence: Number of tasks sampled for each sequence: settings for continual learning
    for idx in epoch_ids:
        behavior_mem_kr, behavior_noise = random.choice(behavior_configs)
        label_mem_kr = random.choice(label_configs)

        print(f"Sampled behavior policy: LTM={behavior_mem_kr},epsilon={behavior_noise}; Sampled reference policy: LTM={label_mem_kr}")
        maze_env = gym.make("mazeworld-v2", enable_render=False, max_steps=max_steps, resolution=(128, 128))

        if(tasks_from_file is not None):
            # Resample the start position and commands sequence from certain tasks
            task = Resampler(random.choice(tasks_from_file))
        else:
            print(n_range)
            task = MazeTaskSampler(n_range=n_range, allow_loops=True, 
                    landmarks_number_range=(6, 10),
                    commands_sequence = 10000,
                    verbose=False)
        task = Resampler(task)

        maze_env.set_task(task)
        results = run_maze_epoch(
                maze_env,
                max_steps,
                label_memkr = label_mem_kr,
                behavior_memkr = behavior_mem_kr,
                behavior_noise = behavior_noise)

        file_path = f'{path_name}/record-{idx:06d}'

        create_directory(file_path)
        numpy.save("%s/observations.npy" % file_path, results["observations"])
        numpy.save("%s/actions_behavior_id.npy" % file_path, results["actions_behavior_id"])
        numpy.save("%s/actions_label_id.npy" % file_path, results["actions_label_id"])
        numpy.save("%s/actions_behavior_val.npy" % file_path, results["actions_behavior_val"])
        numpy.save("%s/actions_label_val.npy" % file_path, results["actions_label_val"])
        numpy.save("%s/commands.npy" % file_path, results["commands"])
        numpy.save("%s/BEVs.npy" % file_path, results["BEVs"])
        numpy.save("%s/rewards.npy" % file_path, results["rewards"])

if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./maze_data/", help="output directory, the data would be stored as output_path/record-xxxx.npy")
    parser.add_argument("--task_source", type=str, choices=['FILE', 'NEW'], help="choose task source to generate the trajectory. FILE: tasks sample from existing file; NEW: create new tasks")
    parser.add_argument("--task_file", type=str, default=None, help="Task source file, used if task_source = FILE")
    parser.add_argument("--max_steps", type=int, default=4000, help="max steps, default:4000")
    parser.add_argument("--n_range", type=str, default="9,21", help="a list of scales separated with comma to randomly choose")
    parser.add_argument('--reference_policy_config', nargs='*', help="List of all LTM ratio, randomly select one for generating reference policy, format: ltm1, ltm2, ...")
    parser.add_argument('--behavior_policy_config', nargs='*', help="List of all LTM ratio and epsilons, randomly select one for generating behavior policy, format ltm1,eps1 ltm2,eps2 ...")
    parser.add_argument("--epochs", type=int, default=1, help="multiple epochs:default:1")
    parser.add_argument("--start_index", type=int, default=0, help="start id of the record number")
    parser.add_argument("--workers", type=int, default=4, help="number of multiprocessing workers")
    args = parser.parse_args()

    if(args.task_source == 'NEW'):
        print("Generating data by sampling new tasks real time")
        tasks_from_file = None
    elif(args.task_source == 'FILE' and args.task_file is not None):
        print("Generating data by sampling from task file: {args.task_file}")
        with open(args.task_file, 'rb') as fr:
            tasks_from_file = pickle.load(fr)
    else:
        raise Exception("Must specify --task_file if task_source == FILE")

    nmin, nmax = args.n_range.split(",")
    n_range = [float(nmin), float(nmax)]

    behavior_configs = []
    label_configs = []

    for val in args.reference_policy_config:
        label_configs.append(float(val))
    for val in args.behavior_policy_config:
        ltm, eps = val.split(",")
        behavior_configs.append((float(ltm), float(eps)))

    worker_splits = args.epochs / args.workers + 1.0e-6
    processes = []
    n_b_t = args.start_index
    for worker_id in range(args.workers):
        n_e_t = n_b_t + worker_splits
        n_b = int(n_b_t)
        n_e = int(n_e_t)

        print("start processes generating %04d to %04d" % (n_b, n_e))
        process = multiprocessing.Process(target=dump_maze, 
                args=(worker_id, args.output_path, range(n_b, n_e), n_range,
                label_configs, behavior_configs,
                args.max_steps, tasks_from_file))
        processes.append(process)
        process.start()

        n_b_t = n_e_t

    for process in processes:
        process.join() 
