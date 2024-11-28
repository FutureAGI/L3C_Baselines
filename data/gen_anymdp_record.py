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
from l3c.anymdp import AnyMDPTaskSampler
from l3c.anymdp import AnyMDPSolverOpt, AnyMDPSolverOTS, AnyMDPSolverQ
from l3c.utils import pseudo_random_seed


class AnyPolicySolver(object):
    """
    Sample Any Policy for AnyMDP
    """
    def __init__(self, env):
        if(not env.task_set):
            raise Exception("AnyMDPEnv is not initialized by 'set_task', must call set_task first")
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        ent_1 = numpy.random.exponential(2.0)
        ent_2 = numpy.random.exponential(1.0e-5)
        self.policy_matrix = numpy.random.normal(size=(self.n_states, self.n_actions), scale=ent_1)
        self.policy_matrix = numpy.exp(self.policy_matrix)
        self.policy_matrix /= numpy.sum(self.policy_matrix, axis=1, keepdims=True)
        self.policy_transfer = numpy.eye(self.n_actions, self.n_actions) + numpy.random.normal(scale=ent_2, size=(self.n_actions, self.n_actions))
        self.policy_transfer = numpy.clip(self.policy_transfer, 0, 1)
        self.policy_transfer = self.policy_transfer / numpy.sum(self.policy_transfer, axis=1, keepdims=True)

    def learner(self, *args, **kwargs):
        self.policy_matrix = numpy.matmul(self.policy_matrix, self.policy_transfer)

    def policy(self, state):
        return numpy.random.choice(self.n_actions, size=1, p=self.policy_matrix[state])[0]


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

    state_list = list()
    lact_list = list()
    bact_list = list()
    reward_list = list()
    
    solveropt = AnyMDPSolverOpt(env)
    solverneg = AnyPolicySolver(env)

    gamma = random.uniform(0.90, 0.99)
    c = 0.005 * random.exponential(1.0)
    alpha = 0.01 * random.exponential(1.0)
    max_steps_q = random.uniform(100, max_steps)
    solverots = AnyMDPSolverOTS(env, 
                                gamma=gamma,
                                c=c,
                                alpha=alpha,
                                max_steps=max_steps_q)

    gamma = random.uniform(0.90, 0.99)
    c = 0.005 * random.exponential(1.0)
    alpha = 0.01 * random.exponential(1.0)
    max_steps_q = random.uniform(100, max_steps)
    solverq = AnyMDPSolverQ(env, 
                                gamma=gamma,
                                c=c,
                                alpha=alpha,
                                max_steps=max_steps_q)

    state, info = env.reset()

    ppl_sum = []
    mse_sum = []

    dstep = 0.05 / (nstate * naction) * (random.random() + 0.01)
    start = min(random.exponential(1.0), 1.0)
    end = random.uniform(0, start)
    
    # Noise Distilling
    random_scheduler = numpy.clip(start - numpy.arange(max_steps + 1) * dstep,
                                 end, 1.0)

    def resample_solver():
        sel = random.random()
        if(sel < 0.05):
            return solveropt
        elif(sel < 0.45):
            return solverots
        elif(sel < 0.75):
            return solverq
        else:
            return solverneg

    solver = resample_solver()
    need_resample = (random.random() > 0.5)
    resample_freq = random.random() * 0.05

    while steps <= max_steps:
        if(offpolicy_labeling):
            if(random.random() < random_scheduler[steps]):
                bact = random.choice(range(naction))
            else:
                bact = solver.policy(state)
            lact = solveropt.policy(state)
        else:
            bact = solverots.policy(state)
            lact = bact

        next_state, reward, done, info = env.step(bact)

        ppl = -numpy.log(info["transition_gt"][next_state])
        mse = (reward - info["reward_gt"]) ** 2
        ppl_sum.append(ppl)
        mse_sum.append(mse)

        solverots.learner(state, bact, next_state, reward, done)
        solverq.learner(state, bact, next_state, reward, done)
        solverneg.learner()

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
            if(random.random() < resample_freq and need_resample):
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
