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
from l3c.anymdp import AnyMDPSolverOpt, AnyMDPSolverOTS
from l3c.utils import pseudo_random_seed

class PolicyScheduler:
    def __init__(self, L,
            opt_start_range=[0.0, 0.0],
            opt_end_range=[1.0, 1.0],
            q_start_range=[1.0, 1.0],
            q_end_range=[0.0, 0.0],
            eps_start_range=[0.5, 1.0],
            eps_end_range=[0.0, 0.5],
            opt_step=1.0e-3,
            q_step=1.0e-3,
            eps_step=1.0e-3):

        def sample(x):
            return random.uniform(x[0], x[1])

        opt_start = sample(opt_start_range)
        opt_end = sample(opt_end_range)
        if(opt_end < opt_start):
            opt_step = -opt_step

        q_start = sample(q_start_range)
        q_end = sample(q_end_range)
        if(q_end < q_start):
            q_step = -q_step

        eps_start = sample(eps_start_range)
        eps_end = sample(eps_end_range)
        if(eps_end < eps_start):
            eps_step = -eps_step

        opt_arr = numpy.clip(numpy.arange(L) * opt_step + opt_start,
                             min(opt_start, opt_end), max(opt_start, opt_end))
        q_arr = numpy.clip(numpy.arange(L) * q_step + q_start,
                             min(q_start, q_end), max(q_start, q_end))
        eps_arr = numpy.clip(numpy.arange(L) * eps_step + eps_start,
                             min(eps_start, eps_end), max(eps_start, eps_end))

        prob = numpy.stack([opt_arr, q_arr, eps_arr], axis=0)
        sum_prob = numpy.clip(numpy.sum(prob, axis=0, keepdims=True), 1.0e-6, None)
        self.prob = prob / sum_prob
        self.prob[2] += 1.0 - numpy.sum(self.prob, axis=0)

    def __call__(self):
        selection = (self.prob.cumsum(0) > numpy.random.rand(self.prob.shape[1])[None, :]).argmax(0)
        return selection

def run_epoch(
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
    
    solver1 = AnyMDPSolverOpt(env)
    solver2 = AnyMDPSolverOTS(env)

    state, info = env.reset()

    ppl_sum = []
    mse_sum = []

    dstep = 0.02 / (nstate * naction)

    ps_b = PolicyScheduler(max_steps + 1,
                            opt_start_range=[-2.0, 0.0],
                            opt_end_range=[-0.5, 0.5],
                            q_start_range=[0.1, 1.0],
                            q_end_range=[0.1, 1.0],
                            eps_start_range=[0.1, 1.0],
                            eps_end_range=[0.1, 1.0],
                            opt_step=dstep,
                            q_step=dstep,
                            eps_step=dstep
                            )
    ps_l = PolicyScheduler(max_steps + 1, 
                            opt_start_range=[-1.0, -1.0],
                            opt_end_range=[0.5, 0.5],
                            q_start_range=[1.0, 1.0],
                            q_end_range=[1.0, 1.0],
                            eps_start_range=[0.0, 0.0],
                            eps_end_range=[0.0, 0.0],
                            opt_step=dstep,
                            q_step=dstep,
                            eps_step=dstep
                            )
    ps_b_traj = ps_b()
    ps_l_traj = ps_l()

    _act_type = ['OPT', 'VI', 'Random']

    def gen_act(state, act_type):
        if(act_type == 0):
            act = solver1.policy(state)
        elif(act_type == 1):
            act = solver2.policy(state)
        else:
            act = random.choice(range(naction))
        return act

    while steps <= max_steps:
        bact = gen_act(state, ps_b_traj[steps - 1]) 
        if(offpolicy_labeling):
            lact = gen_act(state, ps_l_traj[steps - 1])
        else:
            lact = bact

        next_state, reward, done, info = env.step(bact)

        ppl = -numpy.log(env.transition_matrix[state, bact, next_state])
        mse = (reward - env.reward_matrix[state, bact]) ** 2
        ppl_sum.append(ppl)
        mse_sum.append(mse)

        solver2.learner(state, bact, next_state, reward, done)

        if(done):
            next_state, info = env.reset()

        state_list.append(state)
        bact_list.append(bact)
        lact_list.append(lact)
        reward_list.append(reward)

        state = next_state
        steps += 1

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
            task = AnyMDPTaskSampler(nstates, nactions)

        env.set_task(task)
        results = run_epoch(env, max_steps, offpolicy_labeling=is_offpolicy_labeling)

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
                        args.state_num, args.action_num, 
                        (args.offpolicy_labeling>0), args.max_steps, tasks_from_file))
        processes.append(process)
        process.start()

        n_b_t = n_e_t

    for process in processes:
        process.join() 
