#!/usr/bin/env python
# coding=utf8
# File: models.py
import sys
import random
import torch
import numpy
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint  
from modules import Encoder, Decoder, ResBlock, MapDecoder, ActionDecoder, LatentDecoder, VAE
from modules import DiffusionLayers
from .mazeworld_base import MazeModelBase
from utils import ce_loss_mask, mse_loss_mask, img_pro, img_post

class MazeModelXL(MazeModelBase):
    def __init__(self, config): 
        super().__init__(config)
        self.init_mem()
        self.mem_len = config.memory_length

    def init_mem(self):
        self.memory = None
        
    def merge_mem(self, cache):
        if(cache is not None and self.memory is not None):
            new_mem = []
            for mem, ca in zip(self.memory, cache):
                new_mem.append(torch.cat((mem, ca), dim=1))
        elif(self.memory is not None):
            new_mem = self.memory
        elif(cache is not None):
            new_mem = cache
        else:
            new_mem = None
        return new_mem

    def update_mem(self, cache):
        memories = self.merge_mem(cache)
        if(memories[0].shape[1] > 2 * self.mem_len):
            new_mem = []
            for memory in memories:
                new_mem.append(memory[:, -self.mem_len:])
            self.memory = new_mem
        else:
            self.memory = memories
            
    def sequential_loss(self, observations, behavior_actions, label_actions, targets, state_dropout=0.0, reduce='mean'):
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.decformer.requires_grad_(True)
        self.act_decoder.requires_grad_(True)
        self.lat_decoder.requires_grad_(True)

        inputs = img_pro(observations)
        z_rec, z_pred, a_pred, cache = self.forward(inputs[:, :-1], behavior_actions, cache=None, need_cache=True, state_dropout=state_dropout)
        self.update_mem(cache)
        if(self.wm_type == 'image'):
            with torch.no_grad():
                z_rec_l, _ = self.vae(inputs[:, -1:])
                z_rec_l = torch.cat((z_rec, z_rec_l), dim=1)
            if(self.is_diffusion):
                lmse_z = self.lat_decoder.loss_DDPM(z_pred, z_rec_l[:, 1:], mask=self.loss_mask[:, :z_pred.shape[1]], reduce=reduce)
                z_pred = self.lat_decoder(z_rec_l[:, 1:], z_pred)
            else:
                z_pred = self.lat_decoder(z_pred)
                lmse_z = mse_loss_mask(z_pred, z_rec_l[:, 1:], mask=self.loss_mask[:, :z_pred.shape[1]], reduce=reduce)
            obs_pred = self.vae.decoding(z_pred)
            lmse_obs = mse_loss_mask(obs_pred, inputs[:, 1:], mask=self.loss_mask[:, :obs_pred.shape[1]], reduce=reduce)
        elif(self.wm_type == 'target'):
            tx = targets[:, :, 0] * torch.cos(targets[:, :, 1])
            ty = targets[:, :, 0] * torch.sin(targets[:, :, 1])
            targets = torch.stack((tx, ty), dim=2)
            target_pred = self.lat_decoder(z_pred)
            lmse_z = mse_loss_mask(target_pred, targets, mask=self.loss_mask[:, :z_pred.shape[1]], reduce=reduce)
            lmse_obs = torch.tensor(0.0).to(lmse_z.device)

        lce_act = ce_loss_mask(a_pred, label_actions, mask=self.loss_mask[:, :a_pred.shape[1]], reduce=reduce)
        cnt = torch.tensor(label_actions.shape[0] * label_actions.shape[1], dtype=torch.int, device=label_actions.device)

        return lmse_obs, lmse_z, lce_act, cnt

    def inference_step_by_step(self, observations, actions, T, cur_step, device, n_step=1, cache=None, verbose=True):
        lc = 0
        lm = 0
        if(cache is not None):
            lc = cache[0].shape[1]
        if(self.memory is not None):
            lm = self.memory[0].shape[1]
        print(f"cache:{lc}; memory:{lm}")
        pred_obs_list, pred_act_list, valid_cache = super().inference_step_by_step(observations, actions, T, cur_step, device, n_step=n_step, cache=cache, verbose=verbose)

        if(valid_cache[0].shape[1] > self.mem_len):
            self.update_mem(valid_cache)
            valid_cache = None

        return pred_obs_list, pred_act_list, valid_cache

if __name__=="__main__":
    from utils import Configure
    config=Configure()
    config.from_yaml(sys.argv[1])

    model = MazeModelXL(config.model_config)

    observation = torch.randn(8, 33, 3, 128, 128)
    action = torch.randint(4, (8, 32)) 
    reward = torch.randn(8, 32)
    local_map = torch.randn(8, 32, 3, 7, 7)

    vae_loss = model.vae_loss(observation)
    losses = model.sequential_loss(observation, action, action, None)
    rec_img, img_out, act_out, cache = model.inference_step_by_step(observation[:, :11], action[:, :10], 0.02, 0, action.device)
    print("vae:", vae_loss, "sequential:", losses)
    print(img_out[0].shape, act_out.shape)
    print(len(cache))
    print(cache[0].shape)
