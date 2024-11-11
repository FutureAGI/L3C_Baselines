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
from l3c_baselines.utils import weighted_loss, img_pro, img_post
from l3c_baselines.utils import parameters_regularization, count_parameters
from l3c_baselines.utils import log_debug, log_warn, log_fatal
from l3c_baselines.modules import ImageEncoder, ImageDecoder
from .decision_model import RSADecisionModel

class AnyMDPRSA(RSADecisionModel):
    def __init__(self, config, verbose=False): 
        super().__init__(config)

        # Loss weighting
        loss_weight = torch.cat( (torch.linspace(1.0e-3, 1.0, config.context_warmup),
                                  torch.full((config.max_position_loss_weighting - config.context_warmup,), 1.0)
                                  ), 
                                dim=0)
        loss_weight = loss_weight / torch.sum(loss_weight)

        self.register_buffer('loss_weight', loss_weight)
        self.set_train_config(config)

        self.nactions = config.action_dim

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
                            state_dropout=0.0,
                            reward_dropout=0.0,
                            update_memory=True,
                            use_loss_weight=True,
                            reduce_dim=1):
    
        bsz = behavior_actions.shape[0]
        seq_len = behavior_actions.shape[1]
        # Pay attention position must be acquired before calling forward()
        ps = self.causal_model.position
        pe = ps + seq_len

        # Predict the latent representation of action and next frame (World Model)
        s_pred, a_pred, r_pred, _ = self.forward(
                prompts, observations[:, :-1], behavior_actions, rewards,
                cache=None, need_cache=False, state_dropout=state_dropout,
                update_memory=update_memory)

        # Calculate the loss information
        loss = dict()

        # Mask out the invalid actions
        loss_weight = (label_actions.ge(0) * label_actions.lt(self.nactions)).to(self.loss_weight.dtype)
        if(use_loss_weight):
            loss_weight = loss_weight * self.loss_weight[ps:pe].unsqueeze(0)

        # World Model Loss - States and Rewards
        loss["wm-s"], loss["count"] = weighted_loss(s_pred, 
                                     gt=observations[:, 1:], 
                                     loss_type="ce",
                                     loss_wht=loss_weight, 
                                     reduce_dim=reduce_dim,
                                     need_cnt=True)
        loss["wm-r"] = weighted_loss(r_pred, 
                                     gt=rewards.view(*rewards.shape,1), 
                                     loss_type="mse",
                                     loss_wht=loss_weight, 
                                     reduce_dim=reduce_dim)

        # Policy Model
        loss["pm"] = weighted_loss(a_pred, 
                                   gt=label_actions, 
                                   loss_type="ce",
                                   loss_wht=loss_weight, 
                                   reduce_dim=reduce_dim)
        # Entropy Loss
        loss["ent"] = weighted_loss(a_pred, 
                                    loss_type="ent", 
                                    loss_wht=loss_weight,
                                    reduce_dim=reduce_dim)

        loss["causal-l2"] = parameters_regularization(self)

        return loss
        

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
    rec_img, img_out, act_out, cache = model.inference_step_by_step(
            observation[:, :5], action[:, :4], 1.0, 0, observation.device)
    print("vae:", vae_loss, "sequential:", losses)
    print(img_out[0].shape, act_out.shape)
    print(len(cache))
    print(cache[0].shape)
