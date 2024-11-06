import numpy
import torch

def img_pro(observations):
    return observations / 255

def img_post(observations):
    return observations * 255


def rewards2go(rewards, gamma=0.98):
    """
    returns a future moving average of rewards
    """
    rolled_rewards = rewards.clone()
    r2go = rewards.clone()
    n = max(min(50, -1/numpy.log10(gamma)), 0)
    for _ in range(n):
        rolled_rewards = gamma * torch.roll(rolled_rewards, shifts=-1, dims=1)
        r2go += rolled_rewards
    return r2go
