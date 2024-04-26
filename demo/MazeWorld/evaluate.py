import argparse
import sys
import os
import torch
import numpy
import gym
import l3c.mazeworld
import cv2
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader, Dataset

from l3c.mazeworld import MazeTaskSampler
from l3c.mazeworld.agents import SmartSLAMAgent

from dataloader import MazeDataSet
from models import MazeModelBase
from utils import VideoWriter


def postprocess_image(img, scale_factor, texts):
    w, h, c = img.shape
    W = w * scale_factor
    H = h * scale_factor
    img_out = cv2.resize(img, (H, W))
    for pos, text in texts:
        reshaped_pos = (pos[0] * scale_factor, pos[1] * scale_factor)
        img_out = cv2.putText(img_out, text, reshaped_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    return img_out

action_texts = ["STOP", "TURN LEFT", "TURN RIGHT", "BACKWARD", "FORWARD"]
def action_text(action):
    return "Actions:" + action_texts[action]

def reward_smoothing(rewards, kernel_size=32):
    def gaussian_kernel(size=36, sigma=6):
        """ Returns a 1D Gaussian kernel array """
        size = int(size) // 2
        x = numpy.linspace(-size, size, 2*size+1)
        gauss = numpy.exp(-0.5 * (x / sigma) ** 2)
        return gauss / gauss.sum()
    # +0.01 and -0.01 to avoid boundary effects
    data_convolved = numpy.convolve(numpy.array(rewards) + 0.01, gaussian_kernel(), mode='same') - 0.01
    return data_convolved

def model_epoch(maze_env, task, model, device, video_writer=None):
    # Example training loop
    maze_env.set_task(task)
    obs = maze_env.reset()
    obs_arr = [obs]
    act_arr = []
    rew_arr = []

    done = False
    step = 0
    cache = None
    act_tor = None
    while not done:
        step += 1
        obs_tor = torch.from_numpy(numpy.array(obs_arr)).float().to(device).unsqueeze(0)
        obs_tor = obs_tor.permute(0, 1, 4, 2, 3)
        act_tor = torch.from_numpy(numpy.array(act_arr)).long().to(device).unsqueeze(0)
        with torch.no_grad():
            rec_obs, next_obs_list, next_act, cache = model.inference_next(obs_tor, act_tor, cache)
        
        next_action = int(next_act.squeeze().item())
        next_obs_gt, next_rew_gt, done, _ = maze_env.step(next_action)
        loc_map = maze_env.maze_core.get_loc_map(3)
        obs_arr.append(next_obs_gt)
        rew_arr.append(next_rew_gt)
        act_arr.append(next_action)

        next_obs_pred_list = []
        for next_obs in next_obs_list:
            next_obs_pred = next_obs.squeeze().permute(1, 2, 0).cpu().numpy()
            next_obs_pred_list.append(next_obs_pred)

        rec_obs_pred = rec_obs.squeeze().permute(1, 2, 0).cpu().numpy()
        obs_err = numpy.sqrt(numpy.mean((next_obs_pred - next_obs_gt) ** 2))
        obs_err_rec = numpy.sqrt(numpy.mean((rec_obs_pred - obs_arr[-2]) ** 2))
        loc_map = cv2.resize(loc_map, (128, 128), interpolation=cv2.INTER_NEAREST)

        if(video_writer is not None):
            line1 = numpy.transpose(numpy.concatenate([obs_arr[-2], rec_obs_pred, obs_arr[-1], loc_map], axis=0), (1, 0, 2))
            line2 = numpy.transpose(numpy.concatenate(next_obs_pred_list[-4:], axis=0), (1, 0, 2))
            raw_img = numpy.concatenate([line1, line2], axis=0)
            img = postprocess_image(raw_img, 3, (
                ((10, 10), "Observation t"),
                ((20, 64), action_text(act_arr[-1])),
                ((138, 10), "Reconstruction t"),
                ((266, 10), "Observation t + 1"),
                ((394, 10), "Local Map"),
                ((10, 138), "Predict t+1 x_T"),
                ((138, 138), "Predict t+1 x_2T/3"),
                ((266, 138), "Predict t+1 x_T/3"),
                ((394, 138), "Predict t+1 x_0"),
                ))
            
            video_writer.add_image(img)

            print("Step: %d, Reward: %f, Reward Summary: %f, Observation Prediction Error: %f, Reconstruction Error: %f" % (step, next_rew_gt, numpy.sum(rew_arr), obs_err, obs_err_rec))

        sys.stdout.flush()
    return reward_smoothing(rew_arr)


def agent_epoch(maze_env, task):
    # Example training loop
    maze_env.set_task(task)
    observation = maze_env.reset()
    agent = SmartSLAMAgent(maze_env=maze_env, render=True)

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
    parser.add_argument('--output', type=str, default="./videos")
    args = parser.parse_args()

    model = MazeModelBase(image_size=128, map_size=7, action_size=5, max_time_step=args.max_time_step)
    use_gpu = torch.cuda.is_available()
    if(use_gpu):
        device = torch.device(f'cuda:0')
    else:
        device = torch.device('cpu')

    DP(model).load_state_dict(torch.load('%s/model.pth' % args.load_path))
    model = model.to(device)

    reward_model = []
    reward_agent = []

    for idx in range(args.test_epochs):

        maze_env = gym.make("mazeworld-discrete-3D-v1", enable_render=False, max_steps=args.test_time_step, task_type="NAVIGATION", resolution=(128, 128))

        task = MazeTaskSampler(n=args.scale, allow_loops=True, 
                wall_density=args.density,
                landmarks_number=args.n_landmarks,
                landmarks_avg_reward=0.5,
                commands_sequence = 10000,
                verbose=False)

        video_writer = VideoWriter("./videos", "demo", window_size=(1536, 768))
        reward_model.append(model_epoch(maze_env, task, model, device, video_writer))
        maze_env.save_trajectory(f'./videos/traj_model_{idx}.jpg')

        reward_agent.append(agent_epoch(maze_env, task))
        maze_env.save_trajectory(f'./videos/traj_agent_{idx}.jpg')

    video_writer.clear()
    reward_model = numpy.array(reward_model)
    reward_agent = numpy.array(reward_agent)
    reward_model = numpy.mean(numpy.array(reward_model), axis=0)
    reward_agent = numpy.mean(numpy.array(reward_agent), axis=0)
    print("Model:")
    print(numpy.array2string(reward_model, precision=3, separator='\t'))
    print("Agent:")
    print(numpy.array2string(reward_agent, precision=3, separator='\t'))