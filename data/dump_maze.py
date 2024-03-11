#!/usr/bin/env python
# coding=utf8
# File: test.py
import gym
import sys
import os
import l3c.mazeworld
import time
import lmdb
import numpy
import argparse
from l3c.mazeworld import MazeTaskSampler
from l3c.mazeworld.agents import SmartSLAMAgent
from numpy import random

def test_agent_maze(n=15,
        maze_type="Discrete2D", 
        max_steps=1000, 
        task_type="NAVIGATION",
        density=0.40,
        n_landmarks=10,
        r_landmarks=0.40):
    print("\n\n--------\n\nRunning agents on maze_type = %s, task_type = %s, n = %d, steps = %s...\n\n"%(maze_type, task_type, n, max_steps))
    if(maze_type == "Discrete2D"):
        maze_env = gym.make("mazeworld-discrete-2D-v1", enable_render=False, max_steps=max_steps, task_type=task_type)
    elif(maze_type == "Discrete3D"):
        maze_env = gym.make("mazeworld-discrete-3D-v1", enable_render=False, max_steps=max_steps, task_type=task_type)
    elif(maze_type == "Continuous3D"):
        maze_env = gym.make("mazeworld-continuous-3D-v1", enable_render=False, max_steps=max_steps, task_type=task_type)
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

if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="maze_data", help="output file name")
    parser.add_argument("--task_type", type=str, default="NAVIGATION", help="task type")
    parser.add_argument("--maze_type", type=str, default="Discrete3D", help="maze type")
    parser.add_argument("--max_steps", type=int, default=4000, help="max steps")
    parser.add_argument("--n", type=int, default=15, help="n")
    parser.add_argument("--density", type=float, default=0.36, help="density")
    parser.add_argument("--n_landmarks", type=int, default=10, help="n_landmarks")
    args = parser.parse_args()

    observations, actions, rewards, maps = test_agent_maze(
            maze_type=args.maze_type, 
            max_steps=args.max_steps,
            n=args.n, 
            task_type=args.task_type, 
            density=args.density, 
            n_landmarks=args.n_landmarks)

    # Convert observations, actions, and rewards to lmdb format and save file
    # Open the lmdb environment
    create_directory(args.output)
    numpy.save("%s/observations.npy" % args.output, observations)
    numpy.save("%s/actions.npy" % args.output, actions)
    numpy.save("%s/rewards.npy" % args.output, rewards)
    numpy.save("%s/maps.npy" % args.output, maps)
