#!/usr/bin/env python
# coding=utf8
# File: dump_maze.py
import gym
import sys
import os
import l3c.mazeworld
import time
import numpy
import argparse
import multiprocessing
from l3c.mazeworld import MazeTaskSampler
from l3c.mazeworld.agents import SmartSLAMAgent
from numpy import random

def run_maze_epoch(n=15,
        maze_type="Discrete2D", 
        max_steps=1000, 
        task_type="NAVIGATION",
        density=0.40,
        memory_keep_ratio=1.0,
        behavior_epsilon=0.40,
        n_landmarks=10,
        r_landmarks=0.40):
    print("\n\n--------\n\nRunning agents on maze_type = %s, task_type = %s, n = %d, steps = %s...\n\n"%(maze_type, task_type, n, max_steps))
    if(maze_type == "Discrete2D"):
        maze_env = gym.make("mazeworld-discrete-2D-v1", enable_render=False, max_steps=max_steps, task_type=task_type, resolution=(128, 128))
    elif(maze_type == "Discrete3D"):
        maze_env = gym.make("mazeworld-discrete-3D-v1", enable_render=False, max_steps=max_steps, task_type=task_type, resolution=(128, 128))
    elif(maze_type == "Continuous3D"):
        maze_env = gym.make("mazeworld-continuous-3D-v1", enable_render=False, max_steps=max_steps, task_type=task_type, resolution=(128, 128))
    else:
        raise Exception("No such maze world type %s"%task_type)

    task = MazeTaskSampler(n=n, allow_loops=True, 
            wall_density=density,
            landmarks_number=n_landmarks,
            landmarks_avg_reward=r_landmarks,
            commands_sequence = 10000,
            verbose=False)

    maze_env.set_task(task)

    # Must intialize agent after reset
    agent = SmartSLAMAgent(maze_env=maze_env, render=False, memory_keep_ratio=memory_keep_ratio)

    done=False
    observation = maze_env.reset()
    sum_reward = 0
    reward = 0
    observation_list = [observation]
    behavior_action_list = []
    label_action_list = []
    reward_list = []
    map_list = []
    interval = 0

    while not done:
        label_action = agent.step(observation, reward)
        if(random.random() < behavior_epsilon):
            action = random.randint(0,4)
        else:
            action = label_action
        behavior_action_list.append(action)
        label_action_list.append(label_action)

        observation, reward, done, info = maze_env.step(action)
        loc_map = maze_env.maze_core.get_loc_map(map_range=3)
        reward_list.append(reward)
        observation_list.append(observation)
        sum_reward += reward

    print("Finish running, sum reward = %f, steps = %d\n"%(sum_reward, len(observation_list)-1))

    return (numpy.asarray(observation_list, dtype=numpy.uint8), 
        numpy.asarray(behavior_action_list, dtype=numpy.uint8), 
        numpy.asarray(label_action_list, dtype=numpy.uint8), 
        numpy.asarray(reward_list, dtype=numpy.float32))

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def dump_maze(path_name, epoch_ids, n_list, n_landmarks_list, density_list, mem_kr_list, maze_type, max_steps, task_type):
    for idx in epoch_ids:
        n = random.choice(n_list)
        n_landmarks = random.choice(n_landmarks_list)
        density = random.choice(density_list)
        mem_kr = random.choice(mem_kr_list)

        observations, behavior_actions, label_actions, rewards = run_maze_epoch(
                maze_type=maze_type,
                max_steps=max_steps,
                n=n,
                task_type=task_type,
                memory_keep_ratio=mem_kr,
                density=density,
                n_landmarks=n_landmarks)

        file_path = f'{path_name}/record-{idx:04d}'

        # Convert observations, actions, and rewards to lmdb format and save file
        # Open the lmdb environment
        create_directory(file_path)
        numpy.save("%s/observations.npy" % file_path, observations)
        numpy.save("%s/actions_behavior.npy" % file_path, behavior_actions)
        numpy.save("%s/actions_label.npy" % file_path, label_actions)
        numpy.save("%s/rewards.npy" % file_path, rewards)

if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./maze_data/", help="output directory, the data would be stored as output_path/record-xxxx.npy")
    parser.add_argument("--task_type", type=str, default="NAVIGATION", help="task type, NAVIGATION/SURVIVAL, default:NAVIGATION")
    parser.add_argument("--maze_type", type=str, default="Discrete3D", help="maze type, Discrete2D/Discrete3D/Continuous3D, default:Discrete3D")
    parser.add_argument("--max_steps", type=int, default=4000, help="max steps, default:4000")
    parser.add_argument("--scale", type=str, default="9,15,21,25,31,35", help="a list of scales separated with comma to randomly choose")
    parser.add_argument("--density", type=str, default="0.20,0.34,0.36,0.38,0.45", help="density:a list of float")
    parser.add_argument("--landmarks", type=str, default="5,6,7,8,9,10", help="landmarks:a list of number of landmarks")
    parser.add_argument("--memory_keep", type=str, default="0.25, 0.50, 1.0", help="random select memory keep ratio from this list")
    parser.add_argument("--epochs", type=int, default=1, help="multiple epochs:default:1")
    parser.add_argument("--workers", type=int, default=4, help="number of multiprocessing workers")
    args = parser.parse_args()

    density_list = list(map(float, args.density.split(",")))
    n_list = list(map(int, args.scale.split(",")))
    n_landmarks_list = list(map(int, args.landmarks.split(",")))
    memory_kr_list = list(map(float, args.memory_keep.split(",")))

    worker_splits = args.epochs / args.workers + 1.0e-6
    processes = []
    n_b_t = 0
    for worker_id in range(args.workers):
        n_e_t = n_b_t + worker_splits
        n_b = int(n_b_t)
        n_e = int(n_e_t)

        print("start processes generating %04d to %04d" % (n_b, n_e))
        process = multiprocessing.Process(target=dump_maze, 
                args=(args.output_path, range(n_b, n_e), n_list, n_landmarks_list, density_list, memory_kr_list, 
                args.maze_type, args.max_steps, args.task_type))
        processes.append(process)
        process.start()

        n_b_t = n_e_t

    for process in processes:
        process.join() 
