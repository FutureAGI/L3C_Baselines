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
import random as rnd
from numpy import random
import l3c
from packaging import version
assert version.parse(l3c.__version__) >= version.parse('0.2.1.19')

from l3c.anymdp import AnyMDPTaskSampler
from l3c.anymdp import AnyMDPSolverOpt
from l3c.utils import pseudo_random_seed

current_folder = os.path.dirname(os.path.abspath(__file__))
if current_folder not in sys.path:
    sys.path.append(current_folder)
from anymdp_behavior_solver import AnyPolicySolver, AnyMDPOptNoiseDistiller, AnyMDPOTSOpter, AnyMDPQNoiseDistiller, AnyMDPOTSNoiseDistiller



def run_epoch(
        epoch_id,
        env,
        max_steps,
        offpolicy_labeling = True,
        ):
    # Must intialize agent after reset
    steps = 0
    
    # Steps to reset the 
    nstate = env.observation_space.n
    naction = env.action_space.n

    # Data Storage
    state_list = list()
    lact_list = list()
    bact_list = list()
    reward_list = list()
    
    # Referrence Policies
    solveropt = AnyMDPSolverOpt(env)

    # List of Behavior Policies
    solverneg = AnyPolicySolver(env)
    solverots = AnyMDPOTSNoiseDistiller(env, max_steps=max_steps)
    solverq = AnyMDPQNoiseDistiller(env, max_steps=max_steps)
    solverotsopt = AnyMDPOTSOpter(env, solver_opt=solveropt, max_steps=max_steps)
    solveroptnoise = AnyMDPOptNoiseDistiller(env, opt_solver=solveropt)
    behavior_list = [solverneg, solverots, solverq, solverotsopt, solveroptnoise]

    state, info = env.reset()

    ppl_sum = []
    mse_sum = []

    solver = random.choice(behavior_list)
    need_resample = (random.random() > 0.60)
    resample_freq = random.random() * 0.05

    def learn(state, bact, next_state, reward, done):
        for solver in behavior_list:
            solver.learner(state, bact, next_state, reward, done)

    def resample_solver():
        if(need_resample and random.random() < resample_freq):
            return random.choice(behavior_list)
        else:
            return solver

    while steps <= max_steps:
        if(offpolicy_labeling):
            bact = solver.policy(state)
            lact = solveropt.policy(state)
        else:
            bact = solverotsopt.policy(state)
            lact = bact

        next_state, reward, done, info = env.step(bact)

        ppl = -numpy.log(info["transition_gt"][next_state])
        mse = (reward - info["reward_gt"]) ** 2
        ppl_sum.append(ppl)
        mse_sum.append(mse)

        learn(state, bact, next_state, reward, done)

        state_list.append(state)
        bact_list.append(bact)
        lact_list.append(lact)
        reward_list.append(reward)

        if(done): # If done, push the next state, but add a dummy action
            state_list.append(next_state)
            bact_list.append(naction)
            lact_list.append(naction)
            reward_list.append(0.0)
            steps += 1
            next_state, info = env.reset()
            solver = resample_solver()

        state = next_state
        steps += 1

    print("Finish running %06d, sum reward: %f, steps: %d, gt_transition_ppl: %f, gt_reward_mse: %f"%(
            epoch_id, numpy.sum(reward_list), len(state_list)-1, numpy.mean(ppl_sum), numpy.mean(mse_sum)))

    return {
            "states": numpy.array(state_list, dtype=numpy.uint32),
            "actions_behavior": numpy.array(bact_list, dtype=numpy.uint32),
            "actions_label": numpy.array(lact_list, dtype=numpy.uint32),
            "rewards": numpy.array(reward_list, dtype=numpy.float32)
            }

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def dump_anymdp(work_id, world_work, path_name, epoch_ids, nstates, nactions, min_state_space,
        is_offpolicy_labeling,
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
            task = AnyMDPTaskSampler(nstates, nactions, min_state_space)

        env.set_task(task)
        results = run_epoch(idx, env, max_steps, offpolicy_labeling=is_offpolicy_labeling)

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
    parser.add_argument("--min_state_space", type=int, default=16, help="minimum state dim in task, default:8")
    parser.add_argument("--max_steps", type=int, default=4000, help="max steps, default:4000")
    parser.add_argument("--offpolicy_labeling", type=int, default=0, help="enable offpolicy labeling (DAgger), default:False")
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
                args=(worker_id, args.workers, args.output_path, range(n_b, n_e), 
                        args.state_num, args.action_num, args.min_state_space,
                        (args.offpolicy_labeling>0), args.max_steps, tasks_from_file))
        processes.append(process)
        process.start()

        n_b_t = n_e_t

    for process in processes:
        process.join() 
