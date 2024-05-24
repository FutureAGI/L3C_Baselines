import random
import argparse
import sys
import os
import torch
import numpy
import gym
import l3c.mazeworld
import cv2
import pickle
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from collections import namedtuple
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader, Dataset

from l3c.mazeworld import MazeTaskSampler
from l3c.mazeworld.agents import SmartSLAMAgent
from l3c.mazeworld.envs.maze_task import MazeTaskManager

TaskConfig = namedtuple("TaskConfig", ["start", "cell_landmarks", "cell_walls", "cell_texts", 
    "cell_size", "wall_height", "agent_height", "initial_life", "max_life",
    "step_reward", "goal_reward", "landmarks_rewards", "landmarks_coordinates", "landmarks_refresh_interval", "commands_sequence"])

from models import MazeModelBase
from utils import create_folder, VideoWriter
from utils import Configure
from utils import custom_load_model


def postprocess_image(img, scale_factor, texts):
    w, h, c = img.shape
    W = w * scale_factor
    H = h * scale_factor
    img_out = cv2.resize(img, (H, W))
    for pos, text in texts:
        reshaped_pos = (pos[0] * scale_factor, pos[1] * scale_factor)
        img_out = cv2.putText(img_out, text, reshaped_pos, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    return img_out

action_texts = ["STOP", "TURN LEFT", "TURN RIGHT", "BACKWARD", "FORWARD"]
def action_text(action):
    return "Actions:" + action_texts[action]

def string_mean_var(mean, var):
    string=""
    for i in range(mean.shape[0]):
        string += f'{mean[i]}\t{var[i]}\n'
    return string

def reward_smoothing(rewards, kernel_size=192):
    def gaussian_kernel(size=kernel_size, sigma=kernel_size // 6):
        """ Returns a 1D Gaussian kernel array """
        size = int(size) // 2
        x = numpy.linspace(-size, size, 2*size+1)
        gauss = numpy.exp(-0.5 * (x / sigma) ** 2)
        return gauss / gauss.sum()
    # +0.01 and -0.01 to avoid boundary effects
    data_convolved = numpy.convolve(numpy.array(rewards) + 0.01, gaussian_kernel(), mode='same') - 0.01
    return data_convolved

def model_epoch(maze_env, task, model, policy_config, device, video_writer=None, video_text=True):
    # Example training loop
    maze_env.set_task(task)
    obs = maze_env.reset()
    obs_arr = [obs]
    act_arr = []
    rew_arr = []
    acc_rew_arr = []

    agent = SmartSLAMAgent(maze_env=maze_env, render=False)
    next_rew_gt = 0.0

    done = False
    step = 0
    cache = None
    act_tor = None
    while not done:
        step += 1
        obs_tor = torch.from_numpy(numpy.array(obs_arr[-1])).float().to(device).unsqueeze(0)
        obs_tor = obs_tor.permute(0, 3, 1, 2).unsqueeze(1)
        with torch.no_grad():
            rec_obs, next_obs, next_act, cache = model.inference_step_by_step(obs_tor, policy_config, cache=cache)

            if(step > 10):
                z_rec, z_sig = model.vae(obs_tor)
                #print("z_rec", z_rec)
        
        next_action = int(next_act.squeeze().item())

        agent_action = agent.step(obs_arr[-1], next_rew_gt)
        print("agent decision:", agent_action)

        next_obs_gt, next_rew_gt, done, _ = maze_env.step(next_action)
        loc_map = maze_env.maze_core.get_loc_map(3)
        obs_arr.append(next_obs_gt)
        rew_arr.append(next_rew_gt)
        if(len(acc_rew_arr) < 1):
            acc_rew_arr.append(max(0.0, next_rew_gt + 0.01))
        else:
            acc_rew_arr.append(acc_rew_arr[-1] + max(0.0, next_rew_gt + 0.01))
        act_arr.append(next_action)

        rec_obs_pred = rec_obs.squeeze().permute(1, 2, 0).cpu().numpy()
        next_obs_pred = next_obs.squeeze().permute(1, 2, 0).cpu().numpy()

        obs_err = numpy.sqrt(numpy.mean((next_obs_pred - next_obs_gt) ** 2))
        obs_err_rec = numpy.sqrt(numpy.mean((rec_obs_pred - obs_arr[-2]) ** 2))
        loc_map = cv2.resize(loc_map, (128, 128), interpolation=cv2.INTER_NEAREST)

        if(video_writer is not None):
            img = numpy.transpose(numpy.concatenate([obs_arr[-2], rec_obs_pred, obs_arr[-1], next_obs_pred], axis=0), (1, 0, 2))
            if(video_text):
                img = postprocess_image(img, 3, (
                    ((10, 10), "t (Ground Truth)"),
                    ((110, 110), action_text(act_arr[-1])),
                    ((138, 10), "t (VAE Reconstruction)"),
                    ((266, 10), "t+1 (Ground Truth)"),
                    ((388, 10), "t+1 (Predict)"),
                    ))
            
            video_writer.add_image(img)

            print("Step: %d, Reward: %f, Reward Summary: %f, Observation Prediction Error: %f, Reconstruction Error: %f" % (step, next_rew_gt, numpy.sum(rew_arr), obs_err, obs_err_rec))

        sys.stdout.flush()
    return reward_smoothing(rew_arr), acc_rew_arr

def random_epoch(maze_env, task):
    # Example training loop
    maze_env.set_task(task)
    observation = maze_env.reset()

    done = False
    step = 0
    rew_arr = []
    acc_rew_arr = []
    reward = 0
    while not done:
        step += 1
        action = random.randint(0, 4)
        observation, reward, done, _ = maze_env.step(action)
        rew_arr.append(reward)
        if(len(acc_rew_arr) < 1):
            acc_rew_arr.append(max(0.0, reward + 0.01))
        else:
            acc_rew_arr.append(acc_rew_arr[-1] + max(0.0, reward + 0.01))
    return reward_smoothing(rew_arr), acc_rew_arr

def agent_epoch(maze_env, task, mem_kr):
    # Example training loop
    maze_env.set_task(task)
    observation = maze_env.reset()
    agent = SmartSLAMAgent(maze_env=maze_env, render=False, memory_keep_ratio=mem_kr)

    done = False
    step = 0
    rew_arr = []
    acc_rew_arr = []
    reward = 0
    while not done:
        step += 1
        action = agent.step(observation, reward)
        observation, reward, done, _ = maze_env.step(action)
        rew_arr.append(reward)
        if(len(acc_rew_arr) < 1):
            acc_rew_arr.append(max(0.0, reward + 0.01))
        else:
            acc_rew_arr.append(acc_rew_arr[-1] + max(0.0, reward + 0.01))
    return reward_smoothing(rew_arr), acc_rew_arr

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

    if(demo_config.write_task is not None):
        #used for dump tasks only
        maze_config = demo_config.maze_config
        tasks = []
        for idx in range(demo_config.test_epochs):
            task = MazeTaskSampler(n=maze_config.scale, allow_loops=True, 
                    wall_density=maze_config.density,
                    landmarks_number=maze_config.n_landmarks,
                    landmarks_avg_reward=0.5,
                    commands_sequence = 10000,
                    verbose=False)
            task_dict = task._asdict()
            tasks.append(task_dict)
        print(f"Writing tasks to {demo_config.write_task} and quit")
        with open(config.write_task, 'wb') as fw:
            pickle.dump(tasks, fw)
        sys.exit(0)
    elif(demo_config.read_task is not None):
        print(f"Reading tasks from {demo_config.read_task}...")
        tasks = []
        with open(demo_config.read_task, 'rb') as fr:
            task_dicts = pickle.load(fr)
            #print(task_dicts)
            for task_dict in task_dicts:
                task = MazeTaskManager.TaskConfig(**task_dict)
                tasks.append(task)
        create_folder(f'{demo_config.output}')
    else:
        raise Exception("Must set 'demo_config.read_task' if write_task is None")

    run_model = demo_config.run_model
    run_rule = demo_config.run_rule
    run_random = demo_config.run_random

    if(run_model):
        model = MazeModelBase(config.model_config)
        use_gpu = torch.cuda.is_available()
        if(use_gpu):
            device = torch.device(f'cuda:0')
        else:
            device = torch.device('cpu')

        load_model_path = demo_config.model_config.load_model_path
        model = custom_load_model(DP(model), f'{load_model_path}/model.pth', strict_check=False)

        model = model.module.to(device)

        reward_model = []
        acc_reward_model = []

    if(run_rule):
        reward_agent = []
        acc_reward_agent = []

    if(run_random):
        reward_random = []
        acc_reward_random = []

    maze_env = gym.make("mazeworld-discrete-3D-v1", enable_render=False, max_steps=demo_config.time_step, task_type="NAVIGATION", resolution=(128, 128))

    # Go across all tasks and perform evaluation
    for idx, task in enumerate(tasks):
        if(run_model):
            video_writer = VideoWriter(f'{demo_config.output}/{idx}/', "demo", window_size=(1536, 384))
            rewards, acc_rewards = model_epoch(maze_env, task, model, config.demo_config.policy, device, video_writer, video_text=True)
            reward_model.append(rewards)
            acc_reward_model.append(acc_rewards)
            maze_env.save_trajectory(f'{demo_config.output}/{idx}/traj_model_agent.jpg')
            video_writer.clear()

        if(run_rule):
            rewards, acc_rewards = agent_epoch(maze_env, task, demo_config.rule_config.mem_kr)
            reward_agent.append(rewards)
            acc_reward_agent.append(acc_rewards)
            create_folder(f'{demo_config.output}/{idx}')
            maze_env.save_trajectory(f'{demo_config.output}/{idx}/traj_rule_agent.jpg')

        if(run_random):
            rewards, acc_rewards = random_epoch(maze_env, task)
            reward_random.append(rewards)
            acc_reward_random.append(acc_rewards)
            create_folder(f'{demo_config.output}/{idx}')
            maze_env.save_trajectory(f'{demo_config.output}/{idx}/traj_random_agent.jpg')

    if(run_model):
        reward_model = numpy.array(reward_model, dtype=numpy.float64)
        rewards_mean = numpy.mean(reward_model, axis=0)
        rewards_std = numpy.std(reward_model, axis=0)
        acc_rewards_mean = numpy.mean(acc_reward_model, axis=0)
        acc_rewards_std = numpy.std(acc_reward_model, axis=0)
        string_model = string_mean_var(rewards_mean, rewards_std)
        with open(f'{demo_config.output}/reward_model_agent.txt', 'w') as f_model:
            f_model.write(string_model)
        acc_string_model = string_mean_var(acc_rewards_mean, acc_rewards_std)
        with open(f'{demo_config.output}/acc_reward_model_agent.txt', 'w') as f_model:
            f_model.write(acc_string_model)

    if(run_rule):
        reward_agent = numpy.array(reward_agent, dtype=numpy.float64)
        rewards_mean = numpy.mean(reward_agent, axis=0)
        rewards_std = numpy.std(reward_agent, axis=0)
        acc_rewards_mean = numpy.mean(acc_reward_agent, axis=0)
        acc_rewards_std = numpy.std(acc_reward_agent, axis=0)
        string_agent = string_mean_var(rewards_mean, rewards_std)
        with open(f'{demo_config.output}/reward_rule_agent.txt', 'w') as f_model:
            f_model.write(string_agent)
        string_agent = string_mean_var(acc_rewards_mean, acc_rewards_std)
        with open(f'{demo_config.output}/acc_reward_rule_agent.txt', 'w') as f_model:
            f_model.write(string_agent)

    if(run_random):
        reward_random = numpy.array(reward_random, dtype=numpy.float64)
        rewards_mean = numpy.mean(reward_random, axis=0)
        rewards_std = numpy.std(reward_random, axis=0)
        acc_rewards_mean = numpy.mean(acc_reward_random, axis=0)
        acc_rewards_std = numpy.std(acc_reward_random, axis=0)
        string_random = string_mean_var(rewards_mean, rewards_std)
        with open(f'{demo_config.output}/reward_rule_random.txt', 'w') as f_model:
            f_model.write(string_random)
        string_random = string_mean_var(acc_rewards_mean, acc_rewards_std)
        with open(f'{demo_config.output}/acc_reward_rule_random.txt', 'w') as f_model:
            f_model.write(string_random)
