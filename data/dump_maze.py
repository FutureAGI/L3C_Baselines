#!/usr/bin/env python
# coding=utf8
# File: dump_maze.py
import gym
import sys
import os
import l3c.mazeworld
import time
import lmdb
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
    agent = SmartSLAMAgent(maze_env=maze_env, render=False)

    done=False
    observation = maze_env.reset()
    sum_reward = 0
    reward = 0
    observation_list = [observation]
    action_list = []
    reward_list = []
    map_list = []
    interval = 0

    while not done:
        action = agent.step(observation, reward)
        action_list.append(action)
        observation, reward, done, info = maze_env.step(action)
        loc_map = maze_env.maze_core.get_loc_map(map_range=3)
        reward_list.append(reward)
        observation_list.append(observation)
        map_list.append(loc_map)
        sum_reward += reward

    print("Finish running, sum reward = %f, steps = %d\n"%(sum_reward, len(observation_list)-1))

    return (numpy.asarray(observation_list, dtype=numpy.uint8), 
        numpy.asarray(action_list, dtype=numpy.uint8), 
        numpy.asarray(reward_list, dtype=numpy.float32),
        numpy.asarray(map_list, dtype=numpy.uint8))

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def dump_maze(file_name, epoch_ids, n_list, n_landmarks_list, density_list, maze_type, max_steps, task_type):
    for idx in epoch_ids:
        n = random.choice(n_list)
        n_landmarks = random.choice(n_landmarks_list)
        density = random.choice(density_list)

        observations, actions, rewards, maps = run_maze_epoch(
                maze_type=args.maze_type,
                max_steps=args.max_steps,
                n=n,
                task_type=args.task_type,
                density=density,
                n_landmarks=n_landmarks)

        if(args.epochs > 1):
            file_path = "%s-%04d"%(file_name, idx)
        else:
            file_path = file_name

        # Convert observations, actions, and rewards to lmdb format and save file
        # Open the lmdb environment
        create_directory(file_path)
        numpy.save("%s/observations.npy" % file_path, observations)
        numpy.save("%s/actions.npy" % file_path, actions)
        numpy.save("%s/rewards.npy" % file_path, rewards)
        numpy.save("%s/maps.npy" % file_path, maps)

if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="maze_data", help="output file name")
    parser.add_argument("--task_type", type=str, default="NAVIGATION", help="task type")
    parser.add_argument("--maze_type", type=str, default="Discrete3D", help="maze type")
    parser.add_argument("--max_steps", type=int, default=4000, help="max steps")
    parser.add_argument("--n", type=str, default="9,15,25,35", help="n:a list of int")
    parser.add_argument("--density", type=str, default="0.20,0.34,0.36,0.38,0.45", help="density:a list of float")
    parser.add_argument("--n_landmarks", type=str, default="5,6,7,8,9,10", help="n_landmarks:a list of int")
    parser.add_argument("--epochs", type=int, default=1, help="multiple epochs:default:1")
    parser.add_argument("--workers", type=int, default=4, help="number of multiprocessing workers")
    args = parser.parse_args()

    density_list = list(map(float, args.density.split(",")))
    n_list = list(map(int, args.n.split(",")))
    n_landmarks_list = list(map(int, args.n_landmarks.split(",")))

    worker_splits = args.epochs / args.workers + 1.0e-6
    processes = []
    n_b_t = 0
    for worker_id in range(args.workers):
        n_e_t = n_b_t + worker_splits
        n_b = int(n_b_t)
        n_e = int(n_e_t)

        print("start processes generating %04d to %04d" % (n_b, n_e))
        process = multiprocessing.Process(target=dump_maze, 
                args=(args.output, range(n_b, n_e), n_list, n_landmarks_list, density_list, args.maze_type, args.max_steps, args.task_type))
        processes.append(process)
        process.start()

        n_b_t = n_e_t

    for process in processes:
        process.join() 
