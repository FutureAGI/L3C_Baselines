#!/usr/bin/env python
# coding=utf8
# File: models.py
import random
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint  
from modules import Encoder, Decoder, ResBlock, MapDecoder, ActionDecoder, LatentDecoder, VAE
from modules import DecisionTransformer
from modules import DiffusionLayers
from utils import ce_loss_mask, mse_loss_mask, img_pro, img_post

class MazeModelBase2(nn.Module):
    def __init__(self, 
                 image_size=128,
                 action_size=5,
                 map_size=7,
                 latent_size=1024,
                 hidden_size=1024,
                 image_encoder_size=384,
                 nhead=16,
                 max_time_step=1024,
                 n_res_block=2,
                 n_trn_block=24):
        super().__init__()

        self.hidden_size = hidden_size

        self.latent_size = latent_size

        self.encoder = Encoder(image_size, 3, image_encoder_size, n_res_block)

        self.decoder = Decoder(image_size, latent_size, image_encoder_size, 3, n_res_block)

        self.vae = VAE(latent_size, self.encoder, self.decoder) 

        self.decformer = DecisionTransformer(latent_size, action_size, n_trn_block, hidden_size, nhead, max_time_step)

        self.act_decoder = ActionDecoder(hidden_size, 4 * hidden_size, action_size, dropout=0.10)

        self.lat_decoder = LatentDecoder(hidden_size, 2 * hidden_size, hidden_size, dropout=0.10)

        context_warmup = 256
        loss_mask = torch.cat((
                torch.linspace(0.0, 1.0, context_warmup).unsqueeze(0),
                torch.full((1, max_time_step - context_warmup, ), 1.0)), dim=1)
        self.register_buffer('loss_mask', loss_mask)

    def forward(self, observations, actions, cache=None, need_cache=True):
        """
        Input Size:
            observations:[B, NT, C, W, H]
            actions:[B, NT / (NT - 1)] 
            cache: [B, NC, H]
        """
        
        # Encode with VAE
        B, NT = actions.shape
        with torch.no_grad():
            z_rec, _ = self.vae(observations)

        # Temporal Encoders
        z_pred, a_pred, new_cache = self.decformer(z_rec, actions, cache=cache, need_cache=need_cache)

        # Decode Action [B, N_T, action_size]
        a_pred = self.act_decoder(a_pred)
        z_pred = self.lat_decoder(z_pred)

        return z_rec, z_pred, a_pred, new_cache

    def vae_loss(self, observations, _lambda=1.0e-5, _sigma=1.0):
        self.vae.requires_grad_(True)
        self.encoder.requires_grad_(True)
        self.decoder.requires_grad_(True)
        return self.vae.loss(img_pro(observations), _lambda=_lambda, _sigma=_sigma)

    def sequential_loss(self, observations, actions, reduce='mean'):
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.decformer.requires_grad_(True)
        self.act_decoder.requires_grad_(True)
        self.lat_decoder.requires_grad_(True)

        inputs = img_pro(observations)
        z_rec, z_pred, a_pred, cache = self.forward(inputs[:, :-1], actions, cache=None, need_cache=False)
        obs_pred = self.vae.decoding(z_pred)

        lmse_obs = mse_loss_mask(obs_pred, inputs[:, 1:], mask=self.loss_mask[:, :obs_pred.shape[1]], reduce=reduce)
        lce_act = ce_loss_mask(a_pred, actions, mask=self.loss_mask[:, :a_pred.shape[1]], reduce=reduce)
        cnt = torch.tensor(actions.shape[0] * actions.shape[1], dtype=torch.int, device=actions.device)

        return lmse_obs, lce_act, cnt

    def sequential_loss_with_decoding(self, observations, actions, reduce='mean'):
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.decformer.requires_grad_(True)
        self.act_decoder.requires_grad_(True)
        self.lat_decoder.requires_grad_(True)

        inputs = img_pro(observations)
        z_rec, z_pred, a_pred, cache = self.forward(inputs[:, :-1], actions, cache=None, need_cache=False)
        obs_pred = self.vae.decoding(z_pred)

        lmse_obs = mse_loss_mask(obs_pred, inputs[:, 1:], reduce=reduce)
        lce_act = ce_loss_mask(a_pred, actions, reduce=reduce)
        cnt = torch.tensor(actions.shape[0] * actions.shape[1], dtype=torch.int, device=actions.device)

        return lmse_obs, lce_act, cnt

    def inference_step_by_step(self, observation, cache=None):
        """
        Inference a_t, s_{t+1} give s_t and caches infered from s_0, a_0, ..., s_{t-1}, a_{t-1}
        """
        B, _, C, W, H = observation.shape
        device = observation.device
        valid_act = torch.zeros((B, 1), dtype=torch.int).to(device)
        valid_obs = img_pro(observation)

        # Inference Action First
        with torch.no_grad():
            z_rec, z_pred, a_pred, new_cache  = self.forward(valid_obs, valid_act, cache=cache, need_cache=True)
            r_action = torch.multinomial(a_pred[:, -1], num_samples=1).squeeze(1)
            g_action = torch.argmax(a_pred[:, -1], dim=-1, keepdim=False)
            if(random.random() < 0.20):
                flag = "Random"
                valid_act[:, -1] = r_action
            else:
                flag = "Greedy"
                valid_act[:, -1] = g_action
            print("Decision:", a_pred, flag, "random", r_action, "greedy", g_action)

            # Inference Next Observation based on Sampled Action
            z_rec, z_pred, a_pred, new_cache  = self.forward(valid_obs, valid_act, cache=cache, need_cache=True)

            # Decode the prediction
            pred_obs = self.vae.decoding(z_pred)
            rec_obs = self.vae.decoding(z_rec)

        return img_post(rec_obs), img_post(pred_obs), valid_act[:, -1], new_cache
        

if __name__=="__main__":
    model = MazeModelBase2()
    observation = torch.randn(8, 33, 3, 128, 128)
    action = torch.randint(4, (8, 32)) 
    reward = torch.randn(8, 32)
    local_map = torch.randn(8, 32, 3, 7, 7)

    vae_loss = model.vae_loss(observation)
    losses = model.sequential_loss(observation, action)
    rec_img, img_out, act_out, cache = model.inference_next(observation, action)
    print("vae:", vae_loss, "sequential:", losses)
    print(img_out[0].shape, act_out.shape)
    print(len(cache))
    print(cache[0].shape)
