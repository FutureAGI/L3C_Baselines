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
                            reduce='mean'):
    
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
        loss["wm-s"] = ce_loss_mask(s_pred, observations[:, 1:], mask=self.loss_weight[:, ps:pe], reduce=reduce)
        loss["wm-r"] = mse_loss_mask(r_pred, rewards.view(*rewards.shape,1), 
                                    mask=self.loss_weight[:, ps:pe], reduce=reduce)

        # Policy Model and Entropy Loss
        loss["pm"] = ce_loss_mask(a_pred, label_actions, mask=self.loss_weight[:, ps:pe], reduce=reduce)
        loss["ent"] = ent_loss(a_pred, reduce=reduce)
        loss["count"] = torch.tensor(bsz * seq_len, dtype=torch.int, device=a_pred.device)
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
