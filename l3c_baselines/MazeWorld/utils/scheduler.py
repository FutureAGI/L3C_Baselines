import torch
from torch import nn
from torch.nn import functional as F

def img_pro(observations):
    return observations / 255

def img_post(observations):
    return observations * 255

class SigmaScheduler(object):
    def __init__(self, anealing_step):
        self._anealing_step = anealing_step
        self._step = 0

    def step(self):
        self._step += 1

    def __call__(self):
        return min(max(0.0, (self._step / self._anealing_step - 1.0)), 1.0)

def noam_scheduler(it, warmup_steps):
    vit = max(it, 1)
    lr_warm = vit / warmup_steps # warm up steps
    lr_decay = ((vit / warmup_steps) ** (-0.5))
    return min(lr_warm, lr_decay)
