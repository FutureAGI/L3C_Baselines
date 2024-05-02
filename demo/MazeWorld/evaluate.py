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

from dataloader import MazeDataSet
from models import MazeModelBase1, MazeModelBase2
from utils import create_folder, VideoWriter


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

def model_epoch(maze_env, task, model, device, video_writer=None, video_text=True):
    # Example training loop
    maze_env.set_task(task)
    obs = maze_env.reset()
    obs_arr = [obs]
    act_arr = []
    rew_arr = []

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
            rec_obs, next_obs, next_act, cache = model.inference_step_by_step(obs_tor, cache=cache)

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
    return reward_smoothing(rew_arr)

def agent_epoch(maze_env, task, mem_kr):
    # Example training loop
    maze_env.set_task(task)
    observation = maze_env.reset()
    agent = SmartSLAMAgent(maze_env=maze_env, render=False, memory_keep_ratio=mem_kr)

    done = False
    step = 0
    cache = None
    rew_arr = []
    reward = 0
    while not done:
        step += 1
        action = agent.step(observation, reward)
        observation, reward, done, _ = maze_env.step(action)
        rew_arr.append(reward)
    return reward_smoothing(rew_arr)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--max_time_step', type=int, default=1024)
    parser.add_argument('--test_time_step', type=int, default=256)
    parser.add_argument('--scale', type=int, default=25)
    parser.add_argument('--density', type=int, default=0.36)
    parser.add_argument('--test_epochs', type=int, default=1)
    parser.add_argument('--n_landmarks', type=int, default=8)
    parser.add_argument('--run_model', type=int, default=0)
    parser.add_argument('--run_rule', type=int, default=1)
    parser.add_argument('--mem_kr', type=float, default=1.0)
    parser.add_argument('--output', type=str, default="./videos")
    parser.add_argument('--write_task', type=str, default=None)
    parser.add_argument('--read_task', type=str, default=None)
    args = parser.parse_args()

    if(args.write_task is not None):
        #used for dump tasks only
        tasks = []
        for idx in range(args.test_epochs):
            task = MazeTaskSampler(n=args.scale, allow_loops=True, 
                    wall_density=args.density,
                    landmarks_number=args.n_landmarks,
                    landmarks_avg_reward=0.5,
                    commands_sequence = 10000,
                    verbose=False)
            task_dict = task._asdict()
            tasks.append(task_dict)
        print(f"Writing tasks to {args.write_task} and quit")
        with open(args.write_task, 'wb') as fw:
            pickle.dump(tasks, fw)
        sys.exit(0)
    elif(args.read_task is not None):
        print(f"Reading tasks from {args.read_task} and quit")
        tasks = []
        with open(args.read_task, 'rb') as fr:
            task_dicts = pickle.load(fr)
            #print(task_dicts)
            for task_dict in task_dicts:
                task = MazeTaskManager.TaskConfig(**task_dict)
                tasks.append(task)
        create_folder(f'{args.output}')
    else:
        raise Exception("Must set '--read_task' if write_task is None")

    if(args.run_model):
        model = MazeModelBase2(image_size=128, map_size=7, action_size=5, max_time_step=args.max_time_step)
        use_gpu = torch.cuda.is_available()
        if(use_gpu):
            device = torch.device(f'cuda:0')
        else:
            device = torch.device('cpu')

        DP(model).load_state_dict(torch.load('%s/model.pth' % args.load_path))
        model = model.to(device)

        reward_model = []

    if(args.run_rule):
        reward_agent = []

    maze_env = gym.make("mazeworld-discrete-3D-v1", enable_render=False, max_steps=args.test_time_step, task_type="NAVIGATION", resolution=(128, 128))

    # Go across all tasks and perform evaluation
    for idx, task in enumerate(tasks):
        if(args.run_model):
            video_writer = VideoWriter(f'{args.output}/{idx}/', "demo", window_size=(1536, 384))
            reward_model.append(model_epoch(maze_env, task, model, device, video_writer, video_text=True))
            maze_env.save_trajectory(f'{args.output}/{idx}/traj_model_agent.jpg')
            video_writer.clear()

        if(args.run_rule):
            reward_agent.append(agent_epoch(maze_env, task, args.mem_kr))
            create_folder(f'{args.output}/{idx}')
            maze_env.save_trajectory(f'{args.output}/{idx}/traj_rule_agent.jpg')

    if(args.run_model):
        reward_model = numpy.array(reward_model)
        reward_model = numpy.mean(numpy.array(reward_model), axis=0)
        string_model = numpy.array2string(reward_model, precision=3, separator='\n', threshold = numpy.inf).strip('[]')
        with open(f'{args.output}/reward_model_agent.txt', 'w') as f_model:
            f_model.write(string_model)

    if(args.run_rule):
        reward_agent = numpy.array(reward_agent)
        reward_agent = numpy.mean(numpy.array(reward_agent), axis=0)
        string_agent = numpy.array2string(reward_agent, precision=3, separator='\n', threshold = numpy.inf).strip('[]')
        with open(f'{args.output}/reward_rule_agent.txt', 'w') as f_model:
            f_model.write(string_agent)
