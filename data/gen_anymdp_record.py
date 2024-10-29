#!/usr/bin/env python
# coding=utf8
# File: dump_maze.py
import gym
import sys
import os
import time
import numpy
import argparse
import multiprocessing
import pickle
import l3c.mazeworld
import random as rnd
from numpy import random
from l3c.anymdp import AnyMDPTaskSampler, Resampler
from l3c.anymdp import AnyMDPSolverOpt, AnyMDPSolverMBRL, AnyMDPSolverQ

def run_epoch(
        env,
        max_steps,
        avg_steps_reset = 200,
        behavior_noise=(0.0, 0.0),
        behavior_policy=(1.0, 0.0),
        label_policy=(1.0, 0.0)
        ):
    # Must intialize agent after reset
    steps = 0
    acc_steps = 0
    
    # Steps to reset the environment
    def resample_reset_steps(_lambda):
        return int(max(1.0 + 0.30 * random.normal(), 0.20) * _lambda)

    # Steps to reset the 
    cur_reset_steps=resample_reset_steps(avg_steps_reset)
    naction = env.action_space.n

    bnoise = numpy.linspace(behavior_noise[0], behavior_noise[1], max_steps + 1)

    state_list = list()
    lact_list = list()
    bact_list = list()
    reward_list = list()
    
    solver1 = AnyMDPSolverOpt(env)
    solver2 = AnyMDPSolverQ(env)

    state, info = env.reset()

    ppl_sum = []
    mse_sum = []

    while steps < max_steps:
        steps += 1
        acc_steps += 1
        if(acc_steps >= cur_reset_steps):
            next_state, info = env.reset()
            bact = naction
            lact = naction
            acc_steps = 0
            reward = 0.0
        else:
            if(random.random() < bnoise[steps]):
                bact = random.choice(range(naction))
            else:
                eps = random.random()
                if(eps < behavior_policy[0]):
                    bact = solver1.policy(state)
                elif(eps < behavior_policy[0] + behavior_policy[1]):
                    bact = solver2.policy(state)
                else:
                    bact = random.choice(range(naction))

            eps = random.random()
            if(eps < label_policy[0]):
                lact = solver1.policy(state)
            elif(eps < label_policy[0] + label_policy[1]):
                lact = solver2.policy(state)
            else:
                lact = random.choice(range(naction))

            next_state, reward, done, info = env.step(bact)

            ppl = -numpy.log(env.transition_matrix[state, bact, next_state])
            mse = (reward - env.reward_matrix[state, bact]) ** 2
            ppl_sum.append(ppl)
            mse_sum.append(mse)

            try:
                solver1.learner(state, bact, next_state, reward, done)
                solver2.learner(state, bact, next_state, reward, done)
            except Exception as e:
                pass

        state_list.append(state)
        bact_list.append(bact)
        lact_list.append(lact)
        reward_list.append(reward)

        state = next_state

    behavior_noise_decay = random.random() / max_steps
    step = 0

    print("Finish running, sum reward: %f, steps: %d, gt_transition_ppl: %f, gt_reward_mse: %f\n"%(
             numpy.sum(reward_list), len(state_list)-1, numpy.mean(ppl_sum), numpy.mean(mse_sum)))

    return {
            "states": numpy.array(state_list, dtype=numpy.uint32),
            "actions_behavior": numpy.array(bact_list, dtype=numpy.uint32),
            "actions_label": numpy.array(lact_list, dtype=numpy.uint32),
            "rewards": numpy.array(reward_list, dtype=numpy.float32)
            }

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def dump_anymdp(work_id, world_work, path_name, epoch_ids, nstates, nactions,
        behavior_noise,
        behavior_policy,
        label_policy,
        max_steps, tasks_from_file):
    # Tasks in Sequence: Number of tasks sampled for each sequence: settings for continual learning
    tasks_num = None
    if(tasks_from_file is not None):
        tasks_num = len(tasks_from_file)
    for idx in epoch_ids:
        env = gym.make("anymdp-v0", max_steps=max_steps)

        if(tasks_from_file is not None):
            # Resample the start position and commands sequence from certain tasks
            task_id = (work_id + idx * world_work) % tasks_num
            task = tasks_from_file[task_id]
        else:
            task = AnyMDPTaskSampler(nstates, nactions)

        env.set_task(task)
        results = run_epoch(env, max_steps, 
                            behavior_noise=rnd.choice(behavior_noise),
                            behavior_policy=rnd.choice(behavior_policy),
                            label_policy=rnd.choice(label_policy))

        file_path = f'{path_name}/record-{idx:06d}'
        create_directory(file_path)

        numpy.save("%s/observations.npy" % file_path, results["states"])
        numpy.save("%s/actions_behavior.npy" % file_path, results["actions_behavior"])
        numpy.save("%s/actions_label.npy" % file_path, results["actions_label"])
        numpy.save("%s/rewards.npy" % file_path, results["rewards"])

if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./anymdp_data/", help="output directory, the data would be stored as output_path/record-xxxx.npy")
    parser.add_argument("--task_source", type=str, choices=['FILE', 'NEW'], help="choose task source to generate the trajectory. FILE: tasks sample from existing file; NEW: create new tasks")
    parser.add_argument("--task_file", type=str, default=None, help="Task source file, used if task_source = FILE")
    parser.add_argument("--state_num", type=int, default=128, help="state num, default:128")
    parser.add_argument("--action_num", type=int, default=5, help="action num, default:5")
    parser.add_argument("--max_steps", type=int, default=4000, help="max steps, default:4000")
    parser.add_argument('--reference_config', nargs='*', help="List of reference policy: e.g., 0.1,0.3 represent OPT=0.1,Q=0.3")
    parser.add_argument('--behavior_config', nargs='*', help="List of behavior policy: e.g., 0.1,0.3 represent OPT=0.1,Q=0.3")
    parser.add_argument('--behavior_noise_config', nargs='*', 
                help="List of behavior policy noise distilling: e.g., 1.0,-1.0, 1.0, 0.1")
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

    # preprocessing
    behavior_noise = []
    for bnc in args.behavior_noise_config:
        bn_min, bn_max = bnc.split(",")
        behavior_noise.append((float(bn_min), float(bn_max)))

    p_ref = []
    for rc in args.reference_config:
        p_ref.append(list(map(float, rc.split(","))))

    b_ref = []
    for bc in args.behavior_config:
        b_ref.append(list(map(float, bc.split(","))))

    # Data Generation
    worker_splits = args.epochs / args.workers + 1.0e-6
    processes = []
    n_b_t = args.start_index
    for worker_id in range(args.workers):
        n_e_t = n_b_t + worker_splits
        n_b = int(n_b_t)
        n_e = int(n_e_t)

        print("start processes generating %04d to %04d" % (n_b, n_e))
        process = multiprocessing.Process(target=dump_anymdp, 
                args=(worker_id, args.workers, args.output_path, range(n_b, n_e), args.state_num, args.action_num, behavior_noise, 
                      p_ref, b_ref, args.max_steps, tasks_from_file))
        processes.append(process)
        process.start()

        n_b_t = n_e_t

    for process in processes:
        process.join() 
