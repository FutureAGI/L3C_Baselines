import argparse
import sys
import os
import torch
import numpy
import gym
import l3c.mazeworld
from reader import MazeDataSet
from models import MazeModels
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from video_writer import VideoWriter
from l3c.mazeworld import MazeTaskSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader, Dataset

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_path(save_model_path, epoch_id):
    directory_path = '%s/%02d/' % (save_model_path, epoch_id)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return ('%s/model.pth' % directory_path,'%s/optimizer.pth' % directory_path) 

def demo_epoch(maze_env, task, model, device, video_writer):
    # Example training loop
    maze_env.set_task(task)
    obs = maze_env.reset()
    obs_arr = [obs]
    act_arr = []
    rew_arr = []
    rew_sum = 0

    done = False
    step = 0
    while not done:
        step += 1
        obs_tor = torch.from_numpy(numpy.array(obs_arr)).float().to(device).unsqueeze(0)
        act_tor = torch.from_numpy(numpy.array(act_arr)).long().to(device).unsqueeze(0)
        obs_tor = obs_tor.permute(0, 1, 4, 2, 3)
        with torch.no_grad():
            next_obs, next_act, next_rew = model.inference_next(obs_tor, act_tor)
        
        next_action = int(next_act.squeeze().item())
        next_obs_gt, next_rew_gt, done, _ = maze_env.step(next_action)
        obs_arr.append(next_obs_gt)
        rew_arr.append(next_rew_gt)
        act_arr.append(next_action)
        rew_sum += next_rew_gt
        next_obs_pred = next_obs.squeeze().permute(1, 2, 0).cpu().numpy()
        obs_err = numpy.sqrt(numpy.mean((next_obs_pred - next_obs_gt) ** 2))
        video_writer.add_image(numpy.concatenate([next_obs_gt, next_obs_pred], axis=1))
        print("Step: %d, Reward: %f, Reward Summary: %f, Observation Prediction Error: %f" % (step, next_rew_gt, rew_sum, obs_err))

        sys.stdout.flush()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--max_steps', type=int, default=512)
    parser.add_argument('--scale', type=int, default=15)
    parser.add_argument('--density', type=int, default=0.36)
    parser.add_argument('--n_landmarks', type=int, default=8)
    parser.add_argument('--output', type=str, default="./videos")
    args = parser.parse_args()

    maze_env = gym.make("mazeworld-discrete-3D-v1", enable_render=False, max_steps=args.max_steps, task_type="NAVIGATION", resolution=(128, 128))

    task = MazeTaskSampler(n=args.scale, allow_loops=True, 
            wall_density=args.density,
            landmarks_number=args.n_landmarks,
            landmarks_avg_reward=0.5,
            commands_sequence = 10000,
            verbose=False)

    model = MazeModels(image_size=128, map_size=7, action_size=5, max_steps=args.max_steps)
    use_gpu = torch.cuda.is_available()
    if(use_gpu):
        device = torch.device(f'cuda:0')
    else:
        device = torch.device('cpu')

    DP(model).load_state_dict(torch.load('%s/model.pth' % args.load_path))
    model = model.to(device)

    video_writer = VideoWriter("./videos", "demo")
    demo_epoch(maze_env, task, model, device, video_writer)
    video_writer.clear()
