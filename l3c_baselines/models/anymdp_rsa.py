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
from l3c_baselines.utils import ce_loss_mask, mse_loss_mask, ent_loss, img_pro, img_post
from l3c_baselines.utils import parameters_regularization, count_parameters
from l3c_baselines.utils import log_debug, log_warn, log_fatal
from l3c_baselines.modules import ImageEncoder, ImageDecoder
from .decision_model import RSADecisionModel

class AnyMDPRSA(RSADecisionModel):
    def __init__(self, config, verbose=False): 
        super().__init__(config)

        loss_weight = torch.cat((
                    torch.linspace(1.0e-3, 1.0, config.context_warmup).unsqueeze(0),
                    torch.full((1, config.max_position_loss_weighting - config.context_warmup,), 1.0)), dim=1)
        self.register_buffer('loss_weight', loss_weight)
        self.set_train_config(config)

        if(verbose):
            log_debug("RSA Decision Model initialized, total params: {}".format(count_parameters(self)))
            log_debug("Causal Block Parameters： {}".format(count_parameters(self.causal_model)))

    def set_train_config(self, config):
        if(config.has_attr("frozen_modules")):
            if("causal_model" in config.frozen_modules):
               self.causal_model.requires_grad_(False)
            else:
               self.causal_model.requires_grad_(True)

            if("s_encoder" in config.frozen_modules):
                self.s_encoder.requires_grad_(False)
            else:
                self.s_encoder.requires_grad_(True)

            if("a_encoder" in config.frozen_modules):
                self.a_encoder.requires_grad_(False)
            else:
                self.a_encoder.requires_grad_(True)

            if(self.r_included):
                if("r_encoder" in config.frozen_modules):
                    self.r_encoder.requires_grad_(False)
                else:
                    self.r_encoder.requires_grad_(True)

            if(self.p_included):
                if("p_encoder" in config.frozen_modules):
                    self.p_encoder.requires_grad_(False)
                else:
                    self.p_encoder.requires_grad_(True)

            if("s_decoder" in config.frozen_modules):
                self.s_decoder.requires_grad_(False)
            else:
                self.s_decoder.requires_grad_(True)

            if("a_decoder" in config.frozen_modules):
                self.a_decoder.requires_grad_(False)
            else:
                self.a_decoder.requires_grad_(True)

            if("r_decoder" in config.frozen_modules):
                self.r_decoder.requires_grad_(False)
            else:
                self.r_decoder.requires_grad_(True)

    def sequential_loss(self, prompts, 
                            observations, 
                            rewards, 
                            behavior_actions, 
                            label_actions, 
                            additional_info=None, # Kept for passing additional information
                            start_position=0, 
                            state_dropout=0.0,
                            reward_dropout=0.0,
                            update_memory=True,
                            gamma = 0.98,
                            reduce_dim=1):
    
        # Predict the latent representation of action and next frame (World Model)
        s_pred, a_pred, r_pred, _ = self.forward(prompts, observations[:, :-1], behavior_actions, rewards,
                cache=None, need_cache=False, state_dropout=state_dropout,
                update_memory=update_memory)

        # Calculate the loss information
        loss = dict()

        bsz = a_pred.shape[0]
        seq_len = a_pred.shape[1]
        ps, pe = start_position, start_position + seq_len

        # World Model Loss - States and Rewards
        loss["wm-s"] = ce_loss_mask(s_pred, observations[:, 1:], mask=self.loss_weight[:, ps:pe], reduce_dim=reduce_dim)
        loss["wm-r"] = mse_loss_mask(r_pred, rewards.view(*rewards.shape,1), 
                                    mask=self.loss_weight[:, ps:pe], reduce_dim=reduce_dim)

        # Policy Model and Entropy Loss
        loss["pm"] = ce_loss_mask(a_pred, label_actions, mask=self.loss_weight[:, ps:pe], reduce_dim=reduce_dim)
        loss["ent"] = ent_loss(a_pred, reduce_dim=reduce_dim)
        loss["count"] = torch.tensor(bsz * seq_len, dtype=torch.int, device=a_pred.device)
        loss["causal-l2"] = parameters_regularization(self)

        return loss
    
    def inference_step_by_step(self, prompts, 
                            observations, 
                            rewards, 
                            behavior_actions, 
                            temp,
                            new_tasks=False,
                            state_dropout=0.0,
                            device=None,
                            cache=None, 
                            need_cache=False,
                            update_memory=True):
        """
        Given: cache - from s_0, a_0, r_0, ..., s_{tc}, a_{tc}, r_{tc}
               observations: s_{tc}, ... s_{t}
               actions: a_{tc}, ..., a_{t-1}
               rewards: r_{tc}, ..., r_{t-1}
        Returns:
            obs_pred: numpy.array [1, state_dim], s_{t+1} (for easy case, state dim = 1, i.e. 1~128)
            act_pred: numpy.array [1], a_{t}
            r_pred: numpy.array [1], r_{t}
            new_cache: torch.array caches up to s_0, a_0, a_r, ..., s_{t}, a_{t}, a_{t} (Notice not to t+n, as t+1 to t+n are imagined)
        """        
        obss = numpy.array(observations, dtype=numpy.int64)
        acts = numpy.array(behavior_actions, dtype=numpy.int64)
        rews = numpy.array(rewards, dtype=numpy.int64)

        # Nobs, W, H, C = obss.shape # for img
        (Nobs,) = obss.shape # for easy case
        (Nacts,) = acts.shape
        (Nrews,) = rews.shape
        if new_tasks:
            assert Nobs == Nacts + 2 == Nrews + 2
        else:
            assert Nobs == Nacts + 1 == Nrews + 1

        valid_obs = torch.from_numpy(obss).int()
        valid_obs = torch.cat((valid_obs, torch.zeros((1,), dtype=torch.int64)), dim=0).unsqueeze(0).to(device)
        valid_act = None
        valid_rew = None
        if(Nacts < 1):
            valid_act = torch.zeros((1, 1), dtype=torch.int64).to(device)
            valid_rew = torch.zeros((1, 1), dtype=torch.int64).to(device)
        elif(new_tasks):
            valid_act = torch.from_numpy(acts).int()
            valid_act = torch.cat((valid_act, torch.zeros((1,), dtype=torch.int64)), dim=0).unsqueeze(0).to(device)
            init_value = torch.zeros((1, 1), dtype=torch.int64).to(device)
            valid_act = torch.cat((valid_act, init_value), dim=1)
            valid_rew = torch.from_numpy(rews).int()
            valid_rew = torch.cat((valid_rew, torch.zeros((1,), dtype=torch.int64)), dim=0).unsqueeze(0).to(device)
            valid_rew = torch.cat((valid_rew, init_value), dim=1)
        else:
            valid_act = torch.from_numpy(acts).int()
            valid_act = torch.cat((valid_act, torch.zeros((1,), dtype=torch.int64)), dim=0).unsqueeze(0).to(device)
            valid_rew = torch.from_numpy(rews).int()
            valid_rew = torch.cat((valid_rew, torch.zeros((1,), dtype=torch.int64)), dim=0).unsqueeze(0).to(device)

        # Update the cache first
        # Only use ground truth
        if(Nobs > 1):
            with torch.no_grad():
                s_pred, a_pred, r_pred, valid_cache  = self.forward(
                    prompts, 
                    observations[:, :-1], 
                    behavior_actions[:, :-1], 
                    rewards[:, :-1],
                    cache=cache, need_cache=need_cache, state_dropout=state_dropout,
                    T=temp,
                    update_memory=update_memory)
        else:
            valid_cache = cache
        n_obs = observations[:, -1:]
        n_action = behavior_actions[:, -1:]
        n_reward = rewards[:, -1:]
        # Predict the latent representation of action and next frame (World Model)
        o_pred, a_pred, r_pred, new_cache = self.forward(
            prompts, 
            n_obs, 
            n_action, 
            n_reward,
            cache=valid_cache, need_cache=need_cache, state_dropout=state_dropout,
            T=temp,
            update_memory=update_memory)
        # Q: How to sample output?
        # Draw samples randomly according to the probability distribution of the input tensor. The result returned is a tensor containing the sampled values.
        pred_action = torch.multinomial(a_pred[:, -1:], num_samples=1).squeeze(1)
        action_out = pred_action.squeeze(0).cpu().numpy()

        state_out = o_pred.squeeze(1).squeeze(0).cpu().numpy()
        reward_out = r_pred.squeeze(1).squeeze(0).cpu().numpy()

        return state_out, action_out, reward_out, new_cache
        

if __name__=="__main__":
    from utils import Configure
    config=Configure()
    config.from_yaml(sys.argv[1])

    model = AnyMDPRSA(config.model_config)

    observation = torch.randn(8, 33, 3, 128, 128)
    action = torch.randint(4, (8, 32)) 
    reward = torch.randn(8, 32)
    local_map = torch.randn(8, 32, 3, 7, 7)

    vae_loss = model.vae_loss(observation)
    losses = model.sequential_loss(None, observation, reward, action, action)
    # rec_img, img_out, act_out, cache = model.inference_step_by_step(
    #         observation[:, :5], action[:, :4], 1.0, 0, observation.device)
    # print("vae:", vae_loss, "sequential:", losses)
    # print(img_out[0].shape, act_out.shape)
    # print(len(cache))
    # print(cache[0].shape)
