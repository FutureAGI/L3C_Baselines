#!/usr/bin/env python
# coding=utf8
# File: models.py
import sys
import random
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint  
from modules import Encoder, Decoder, ResBlock, MapDecoder, ActionDecoder, LatentDecoder, VAE
from modules import DecisionTransformer
from modules import DiffusionLayers
from utils import ce_loss_mask, mse_loss_mask, img_pro, img_post

class MazeModelBase(nn.Module):
    def __init__(self, config): 
        super().__init__()

        self.hidden_size = config.transformer_hidden_size
        self.latent_size = config.image_latent_size
        self.action_size = config.action_size
        context_warmup = config.loss_context_warmup
        if(hasattr(config, "transformer_checkpoints_density")):
            checkpoints_density = config.transformer_checkpoints_density
        else:
            checkpoints_density = -1

        self.encoder = Encoder(config.image_size, 3, config.image_encoder_size, config.n_residual_block)

        self.decoder = Decoder(config.image_size, self.latent_size, config.image_encoder_size, 3, config.n_residual_block)

        self.vae = VAE(self.latent_size, self.encoder, self.decoder) 

        self.decformer = DecisionTransformer(
                self.latent_size, self.action_size, config.n_transformer_block, 
                self.hidden_size, config.transformer_nhead, config.max_time_step, checkpoints_density=checkpoints_density)

        self.act_decoder = ActionDecoder(self.hidden_size, 4 * self.hidden_size, self.action_size, dropout=0.10)

        loss_mask = torch.cat((
                torch.linspace(0.0, 1.0, context_warmup).unsqueeze(0),
                torch.full((1, config.max_time_step - context_warmup, ), 1.0)), dim=1)
        self.register_buffer('loss_mask', loss_mask)

        if(not hasattr(config, "image_decoder_type")):
            raise Exception("image decoder type is not specified, regression / diffusion")
        elif(config.image_decoder_type.lower() == "regression"):
            self.is_diffusion = False
            self.lat_decoder = LatentDecoder(
                self.hidden_size, 
                2 * self.hidden_size, 
                self.hidden_size, dropout=0.0)
        elif(config.image_decoder_type.lower() == "diffusion"):
            self.is_diffusion = True
            self.lat_decoder = DiffusionLayers(
                config.image_decoder.diffusion_steps, # T
                self.latent_size, # hidden size
                self.hidden_size, # condition size
                2 * self.hidden_size, # inner hidden size
            )
        else:
            raise Exception("Unrecognized type", config.image_decoder_type)

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

        return z_rec, z_pred, a_pred, new_cache

    def vae_loss(self, observations, _lambda=1.0e-5, _sigma=1.0):
        self.vae.requires_grad_(True)
        self.encoder.requires_grad_(True)
        self.decoder.requires_grad_(True)
        return self.vae.loss(img_pro(observations), _lambda=_lambda, _sigma=_sigma)

    def sequential_loss(self, observations, behavior_actions, label_actions, reduce='mean'):
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.decformer.requires_grad_(True)
        self.act_decoder.requires_grad_(True)
        self.lat_decoder.requires_grad_(True)

        inputs = img_pro(observations)
        z_rec, z_pred, a_pred, cache = self.forward(inputs[:, :-1], behavior_actions, cache=None, need_cache=False)
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
        lce_act = ce_loss_mask(a_pred, label_actions, mask=self.loss_mask[:, :a_pred.shape[1]], reduce=reduce)
        cnt = torch.tensor(label_actions.shape[0] * label_actions.shape[1], dtype=torch.int, device=label_actions.device)

        return lmse_obs, lmse_z, lce_act, cnt

    def inference_step_by_step(self, observation, config, cache=None, verbose=False):
        """
        Inference a_t, s_{t+1} give s_t and caches infered from s_0, a_0, ..., s_{t-1}, a_{t-1}
        """
        B, _, C, W, H = observation.shape
        device = observation.device
        valid_act = torch.zeros((B, 1), dtype=torch.int).to(device)
        valid_obs = img_pro(observation)

        e_s = config.softmax
        e_g = config.greedy

        # Inference Action First
        with torch.no_grad():
            z_rec, z_pred, a_pred, new_cache  = self.forward(valid_obs, valid_act, cache=cache, need_cache=True)
            s_action = torch.multinomial(a_pred[:, -1], num_samples=1).squeeze(1)
            g_action = torch.argmax(a_pred[:, -1], dim=-1, keepdim=False)
            eps = random.random()
            if(eps < e_s):
                flag = "S"
                valid_act[:, -1] = s_action
            elif(eps < e_s + e_g):
                flag = "G"
                valid_act[:, -1] = g_action
            else:
                flag = "R"
                valid_act[:, -1] = torch.randint_like(r_action, high=self.action_size)
            if(verbose):
                print("Action: {valid_act[:, -1]}\tRaw Output: {a_pred[:, -1]}\tDecision_Flag: {flag}")

            # Inference Next Observation based on Sampled Action
            z_rec, z_pred, a_pred, new_cache  = self.forward(valid_obs, valid_act, cache=cache, need_cache=True)

            # Decode the prediction
            if(self.is_diffusion):
                # Latent Diffusion
                lat_pred_list = self.lat_decoder.inference(z_pred)
                pred_obs = []
                for lat_pred in lat_pred_list:
                    pred_obs.append(img_post(self.vae.decoding(lat_pred)))
            else:
                lat_pred = self.lat_decoder(z_pred)
                pred_obs = self.vae.decoding(lat_pred)

            # Image Decoding
            rec_obs = self.vae.decoding(z_rec)

        return img_post(rec_obs), img_post(pred_obs), valid_act[:, -1], new_cache
        

if __name__=="__main__":
    from utils import Configure
    config=Configure()
    config.from_yaml(sys.argv[1])

    model = MazeModelBase(config.model_config)

    observation = torch.randn(8, 33, 3, 128, 128)
    action = torch.randint(4, (8, 32)) 
    reward = torch.randn(8, 32)
    local_map = torch.randn(8, 32, 3, 7, 7)

    vae_loss = model.vae_loss(observation)
    losses = model.sequential_loss(observation, action)
    rec_img, img_out, act_out, cache = model.inference_step_by_step(observation[:, :1], config.demo_config.policy)
    print("vae:", vae_loss, "sequential:", losses)
    print(img_out[0].shape, act_out.shape)
    print(len(cache))
    print(cache[0].shape)
