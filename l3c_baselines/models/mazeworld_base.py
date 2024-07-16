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
from modules import CausalDecisionModel
from modules import DiffusionLayers
from utils import ce_loss_mask, mse_loss_mask, img_pro, img_post, parameters_regularization

class MazeModelBase(nn.Module):
    def __init__(self, config): 
        super().__init__()

        self.hidden_size = config.transformer_hidden_size
        self.latent_size = config.image_latent_size
        self.action_size = config.action_size
        self.causal_modeling = config.causal_modeling
        context_warmup = config.loss_context_warmup
        if(hasattr(config, "transformer_checkpoints_density")):
            checkpoints_density = config.transformer_checkpoints_density
        else:
            checkpoints_density = -1

        self.encoder = Encoder(config.image_size, 3, config.image_encoder_size, config.n_residual_block)

        self.decoder = Decoder(config.image_size, self.latent_size, config.image_encoder_size, 3, config.n_residual_block)

        self.vae = VAE(self.latent_size, self.encoder, self.decoder) 

        self.decformer = CausalDecisionModel(
                self.latent_size, self.action_size, config.n_transformer_block, 
                self.hidden_size, config.transformer_nhead, config.max_time_step, 
                checkpoints_density=checkpoints_density,
                context_window=config.context_window, 
                model_type = config.causal_modeling)

        self.act_decoder = ActionDecoder(self.hidden_size, 2 * self.hidden_size, self.action_size, dropout=0.0)
        self.wm_type = config.worldmodel_type

        loss_mask = torch.cat((
                torch.linspace(0.0, 1.0, context_warmup).unsqueeze(0),
                torch.full((1, config.max_time_step - context_warmup,), 1.0)), dim=1)
        self.register_buffer('loss_mask', loss_mask)

        if(self.wm_type=='image'):
            if(not hasattr(config, "image_decoder_type")):
                raise Exception("image decoder type is not specified, regression / diffusion")
            elif(config.image_decoder_type.lower() == "regression"):
                self.is_diffusion = False
                self.lat_decoder = LatentDecoder(
                    self.hidden_size, 
                    2 * self.hidden_size, 
                    self.latent_size, dropout=0.0)
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
        elif(self.wm_type=='target'):
            self.lat_decoder = LatentDecoder(
                self.hidden_size,
                2 * self.hidden_size,
                2, dropout=0.0)
        else:
            raise Exception("Unrecognized world model type", config.wm_type)
            
    def forward(self, observations, actions, cache=None, need_cache=True, state_dropout=0.0):
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
        z_pred, a_pred, new_cache = self.decformer(z_rec, actions, cache=cache, need_cache=need_cache, state_dropout=state_dropout)

        # Decode Action [B, N_T, action_size]
        a_pred = self.act_decoder(a_pred)

        return z_rec, z_pred, a_pred, new_cache

    def vae_loss(self, observations, _lambda=1.0e-5, _sigma=1.0):
        self.vae.requires_grad_(True)
        self.encoder.requires_grad_(True)
        self.decoder.requires_grad_(True)
        return self.vae.loss(img_pro(observations), _lambda=_lambda, _sigma=_sigma)

    def sequential_loss(self, observations, behavior_actions, label_actions, targets, state_dropout=0.0, reduce='mean'):
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.decformer.requires_grad_(True)
        self.act_decoder.requires_grad_(True)
        self.lat_decoder.requires_grad_(True)

        inputs = img_pro(observations)
        z_rec, z_pred, a_pred, cache = self.forward(inputs[:, :-1], behavior_actions, cache=None, need_cache=False, state_dropout=state_dropout)
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

    def causal_l2(self):
        return parameters_regularization(self.decformer, self.act_decoder, self.lat_decoder)
        

    def inference_step_by_step(self, observations, actions, T, cur_step, device, n_step=1, cache=None, verbose=True):
        """
        Given: cache - from s_0, a_0, ..., s_{tc}, a_{tc}
               observations: s_{tc}, ... s_{t}
               actions: a_{tc}, ..., a_{t-1}
               T: temperature
        Returns:
            obs_pred: numpy.array [n, W, H, C], s_{t+1}, ..., s_{t+n}
            act_pred: numpy.array [n], a_{t}, ..., a_{t+n-1}
            new_cache: torch.array caches up to s_0, a_0, ..., s_{t}, a_{t} (Notice not to t+n, as t+1 to t+n are imagined)
        """
        obss = numpy.array(observations, dtype=numpy.float32)
        acts = numpy.array(actions, dtype=numpy.int64)
        Nobs, W, H, C = obss.shape
        (No,) = acts.shape

        assert Nobs == No + 1

        valid_obs = torch.from_numpy(img_pro(obss)).float().to(device)
        valid_obs = valid_obs.permute(0, 3, 1, 2).unsqueeze(0)

        if(No < 1):
            valid_act = torch.zeros((1, 1), dtype=torch.int64).to(device)
        else:
            valid_act = torch.from_numpy(acts).int()
            valid_act = torch.cat((valid_act, torch.zeros((1,), dtype=torch.int64)), dim=0).unsqueeze(0).to(device)

        # Update the cache first
        # Only use ground truth
        if(Nobs > 1):
            with torch.no_grad():
                z_rec, z_pred, a_pred, valid_cache  = self.forward(valid_obs[:, :-1], valid_act[:, :-1], cache=cache, need_cache=True)
        else:
            valid_cache = cache

        # Inference Action First
        pred_obs_list = []
        pred_act_list = []
        updated_cache = valid_cache
        n_act = valid_act[:, -1:]
        z_rec, _ = self.vae(valid_obs[:, -1:])

        for step in range(n_step):
            with torch.no_grad():
                # Temporal Encoders
                z_pred, a_pred, _ = self.decformer(z_rec, n_act, cache=updated_cache, need_cache=True)
                # Decode Action [B, N_T, action_size]
                a_pred = self.act_decoder(a_pred, T=T)

                action = torch.multinomial(a_pred[:, 0], num_samples=1).squeeze(1)

                n_act[:, 0] = action
                pred_act_list.append(action.squeeze(0).cpu().numpy())
                if(verbose):
                    print(f"Action: {valid_act[:, -1]} Raw Output: {a_pred[:, -1]}")

                # Inference Next Observation based on Sampled Action
                z_pred, a_pred, updated_cache  = self.decformer(z_rec, n_act, cache=updated_cache, need_cache=True)

                # Only the first step uses the ground truth
                if(step == 0):
                    valid_cache = updated_cache

                # Decode the prediction
                if(self.wm_type=='image'):
                    if(self.is_diffusion):
                        # Latent Diffusion
                        lat_pred_list = self.lat_decoder.inference(z_pred)
                        pred_obs = []
                        for lat_pred in lat_pred_list:
                            pred_obs.append(img_post(self.vae.decoding(lat_pred)))
                    else:
                        lat_pred = self.lat_decoder(z_pred)
                        pred_obs = self.vae.decoding(lat_pred)
                    pred_obs = img_post(pred_obs)
                elif(self.wm_type=='target'):
                    pred_obs = self.lat_decoder(z_pred)

                pred_obs_list.append(pred_obs.squeeze(1).squeeze(0).permute(1, 2, 0).cpu().numpy())

                # Do auto-regression for n_step
                z_rec = self.lat_decoder(z_pred)

        return pred_obs_list, pred_act_list, valid_cache
        

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
