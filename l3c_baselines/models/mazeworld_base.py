#!/usr/bin/env python
# coding=utf8
# File: models.py
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint  
from modules import Encoder, Decoder, ResBlock, MapDecoder, VAE
from modules import DecisionTransformer
from utils import ce_loss_mask, img_pro, img_post

class MazeModelBase(nn.Module):
    def __init__(self, 
                 image_size=128,
                 action_size=5,
                 map_size=7,
                 latent_size=4,
                 hidden_size=512,
                 nhead=16,
                 max_time_step=1024,
                 n_res_block=2,
                 n_trn_block=12):
        super().__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.encoder = Encoder(image_size, 3, hidden_size, n_res_block)

        self.decoder = Decoder(image_size, latent_size, hidden_size, 3, n_res_block)

        self.vae = VAE(latent_size, self.encoder, self.decoder) 

        self.decformer = DecisionTransformer(latent_size, action_size, n_trn_block, hidden_size, nhead, max_time_step)

        self.act_decoder = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, action_size),
            nn.Softmax()
        )

        self.z_decoder = nn.Linear(hidden_size, latent_size)

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
        obs_output, act_output, new_cache = checkpoint(lambda x:self.decformer(x[0], x[1], cache=cache, need_cache=need_cache), (z_rec, actions))

        # Decode Observation [B, N_T, C, W, H]
        # Pass the gradients without adjust the weights of self.decoder
        z_pred = self.z_decoder(obs_output)
        pred_obs = self.decoder(z_pred.reshape(B * NT, -1))
        _, n_c, n_w, n_h = pred_obs.shape
        pred_obs = pred_obs.reshape(B, NT, n_c, n_w, n_h)

        # Decode Action [B, N_T, action_size], without softmax!
        pred_act = self.act_decoder(act_output)

        return z_rec, z_pred, pred_obs, pred_act, new_cache

    def vae_loss(self, observations, _lambda=1.0e-5, _sigma=1.0):
        self.vae.requires_grad_(True)
        self.encoder.requires_grad_(True)
        self.decoder.requires_grad_(True)
        return self.vae.loss(img_pro(observations), _lambda=_lambda, _sigma=_sigma)

    def sequential_loss(self, observations, actions, _lambda=0.01):
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.decformer.requires_grad_(True)
        self.act_decoder.requires_grad_(True)
        self.z_decoder.requires_grad_(True)

        inputs = img_pro(observations)
        z_rec, z_pred, pred_obs, pred_act, cache = self.forward(inputs[:, :-1], actions, cache=None, need_cache=False)

        lmse_obs = _lambda * F.mse_loss(pred_obs, inputs[:, 1:]) + F.mse_loss(z_rec[:, 1:], z_pred[:, :-1])
        lce_act = ce_loss_mask(pred_act, actions)
        #lmse_map = F.mse_loss(pred_map, img_pro(local_maps))
        #lmse_rew = F.mse_loss(rewards, pred_rew.squeeze(-1))
        cnt = torch.tensor(actions.shape[0] * actions.shape[1], dtype=torch.int, device=actions.device)
        return lmse_obs, lce_act, cnt

    def inference_next(self, observations, actions, cache=None):
        """
        Inference a_t, s_{t+1} give s_0, a_0, ..., s_t, a_t 
        """
        B, NT, C, W, H = observations.shape
        device = observations.device
        add_act = torch.zeros((B, 1), dtype=torch.int).to(device)
        add_obs = torch.zeros((B, 1, C, W, H), dtype=torch.float).to(device)

        if(NT < 2):
            valid_act = add_act
        else:
            valid_act = torch.cat([actions, add_act], dim=1)

        if(cache is not None):
            l_cached = cache[0].shape[1] // 2
            valid_obs = observations[:, l_cached:]
            valid_act = valid_act[:, l_cached:]
        else:
            valid_obs = observations
        valid_obs = img_pro(valid_obs)
        B, NT, C, W, H = valid_obs.shape

        # Inference Action First
        with torch.no_grad():
            z_rec, z_pred, pred_obs, pred_act, new_cache  = self.forward(valid_obs, valid_act, cache=cache, need_cache=True)
            n_action = torch.multinomial(pred_act[:, -1], num_samples=1).squeeze(1)
            valid_act[:, -1] = n_action

            # Inference Next Observation based on Sampled Action
            z_rec, z_pred, pred_obs, pred_act, new_cache  = self.forward(valid_obs, valid_act, cache=cache, need_cache=True)
            rec_obs = self.vae.decoding(z_rec)

        return img_post(rec_obs), img_post(pred_obs), n_action, new_cache
        

if __name__=="__main__":
    model = MazeModelBase()
    observation = torch.randn(8, 33, 3, 128, 128)
    action = torch.randint(4, (8, 32)) 
    reward = torch.randn(8, 32)
    local_map = torch.randn(8, 32, 3, 7, 7)

    vae_loss = model.vae_loss(observation)
    losses = model.sequential_loss(observation, action, reward, local_map)
    rec_img, img_out, act_out, cache = model.inference_next(observation, action)
    print("vae:", vae_loss, "sequential:", losses)
    print(img_out.shape, act_out.shape, len(cache), cache[0].shape)
