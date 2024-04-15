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
from l3c.mazeworld import MazeTaskSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader, Dataset

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
        rew_sum += next_rew_gt

        next_obs_pred_list = []
        for next_obs in next_obs_list:
            next_obs_pred = next_obs.squeeze().permute(1, 2, 0).cpu().numpy()
            next_obs_pred_list.append(next_obs_pred)

        rec_obs_pred = rec_obs.squeeze().permute(1, 2, 0).cpu().numpy()
        obs_err = numpy.sqrt(numpy.mean((next_obs_pred - next_obs_gt) ** 2))
        obs_err_rec = numpy.sqrt(numpy.mean((rec_obs_pred - obs_arr[-2]) ** 2))
        loc_map = cv2.resize(loc_map, (128, 128), interpolation=cv2.INTER_NEAREST)

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

        print("Step: %d, Reward: %f, Reward Summary: %f, Observation Prediction Error: %f, Reconstruction Error: %f" % (step, next_rew_gt, rew_sum, obs_err, obs_err_rec))

        sys.stdout.flush()

    obs_tor = torch.from_numpy(numpy.array(obs_arr)).float().to(device).unsqueeze(0)
    act_tor = torch.from_numpy(numpy.array(act_arr)).long().to(device).unsqueeze(0)
    obs_tor = obs_tor.permute(0, 1, 4, 2, 3)
    with torch.no_grad():
        rec_obs, next_obs_list, next_act, cache = model.inference_next(obs_tor, act_tor)
        next_action = int(next_act.squeeze().item())
    print(next_action)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--max_time_step', type=int, default=1024)
    parser.add_argument('--test_time_step', type=int, default=256)
    parser.add_argument('--scale', type=int, default=15)
    parser.add_argument('--density', type=int, default=0.36)
    parser.add_argument('--n_landmarks', type=int, default=8)
    parser.add_argument('--output', type=str, default="./videos")
    args = parser.parse_args()

    maze_env = gym.make("mazeworld-discrete-3D-v1", enable_render=False, max_steps=args.test_time_step, task_type="NAVIGATION", resolution=(128, 128))

    task = MazeTaskSampler(n=args.scale, allow_loops=True, 
            wall_density=args.density,
            landmarks_number=args.n_landmarks,
            landmarks_avg_reward=0.5,
            commands_sequence = 10000,
            verbose=False)

    model = MazeModelBase(image_size=128, map_size=7, action_size=5, max_time_step=args.max_time_step)
    use_gpu = torch.cuda.is_available()
    if(use_gpu):
        device = torch.device(f'cuda:0')
    else:
        device = torch.device('cpu')

    DP(model).load_state_dict(torch.load('%s/model.pth' % args.load_path))
    model = model.to(device)

    video_writer = VideoWriter("./videos", "demo", window_size=(1536, 768))
    demo_epoch(maze_env, task, model, device, video_writer)
    video_writer.clear()
