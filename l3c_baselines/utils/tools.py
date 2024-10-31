import os
import re
import sys
import yaml
import numpy
import torch
from torch import nn
from types import SimpleNamespace
from copy import deepcopy
from dateutil.parser import parse
from collections import defaultdict
from l3c_baselines.utils import log_debug, log_warn, log_fatal

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parameters_regularization(*layers):
    norm = 0
    cnt = 0
    for layer in layers:
        for p in layer.parameters():
            if(p.requires_grad):
                norm += (p ** 2).sum()
                cnt += p.numel()
    return norm / cnt

def format_cache(cache, prefix=''):
    if(cache is None):
        return prefix + ' None'
    elif(isinstance(cache, numpy.ndarray) or isinstance(cache, torch.Tensor)):
        return prefix + ' ' + str(cache.shape)
    elif(isinstance(cache, list) or isinstance(cache, tuple)):
        ret_str = prefix + f'List of length {len(cache)}:\n['
        for subc in cache[:-1]:
            ret_str += format_cache(subc, prefix + ' -')
            ret_str += '\n'
        ret_str += format_cache(cache[-1], prefix + ' -')
        ret_str += ']'
        return ret_str
    else:
        return prefix + ' ' + str(type(cache))

def memory_cpy(cache):
    if(cache is None):
        return None
    elif(isinstance(cache, torch.Tensor)):
        return cache.detach().clone()
    elif(isinstance(cache, list)):
        return [memory_cpy(c) for c in cache]
    elif(isinstance(cache, tuple)):
        return tuple([memory_cpy for c in cache])
    elif(hasattr(cache, 'clone')):
        return cache.clone()
    else:
        return cache

def model_path(save_model_path, epoch_id):
    directory_path = '%s/%02d/' % (save_model_path, epoch_id)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return (f'{directory_path}/model.pth', f'{directory_path}/vae_optimizer.pth', f'{directory_path}/seq_optimizer.pth') 


def gradient_failsafe(model, optimizer, scaler):
    overflow=False
    for param in model.parameters():
        if param.grad is not None and (torch.isinf(param.grad).any() or torch.isnan(param.grad).any()):
            log_warn("gradient contains inf or nan, setting those gradients to zero.")
            overflow=True
    if(overflow):
        for param in model.parameters():
            param.grad.zero_()
        scaler.unscale_(optimizer)
        optimizer.__setstate__({'state': defaultdict(dict)})

def img_pro(observations):
    return observations / 255

def img_post(observations):
    return observations * 255

def print_memory(info="Default"):
    print(info, "Memory allocated:", torch.cuda.memory_allocated(), "Memory cached:", torch.cuda.memory_cached())

def custom_load_model(model, state_dict_path, black_list=[], max_norm_allowed=1.0e+2, strict_check=False, verbose=False):  
    """
    In case of hard condition, shape mismatch, nan/inf are not allowed, which directly lead to failure
    """
    saved_state_dict = torch.load(state_dict_path)  
      
    model_state_dict = model.state_dict()  
      
    matched_state_dict = {}  
    #print("Verbose Model Parameters List:", model_state_dict.keys())

    for param_name, param_tensor in saved_state_dict.items():  
        if param_name in model_state_dict:  
            model_param_shape = model_state_dict[param_name].shape  
              
            is_nan = torch.isnan(param_tensor).any()
            is_inf = torch.isinf(param_tensor).any()
            l2_norm = torch.norm(param_tensor, p=2).item()
            norm_valid = (l2_norm < max_norm_allowed)

            hits_black = False
            for name in black_list:
                if(param_name.find(name) > -1):
                    log_debug(f"Parameter hits {param_name} black lists {name}", on=verbose)
                    hits_black = True
            if(hits_black):
                continue

            if model_param_shape == param_tensor.shape and (not is_nan):  
                if(not norm_valid):
                    log_warn(f"Large norm ({l2_norm}) encountered in parameter {param_name}. Keep Loading...", on=verbose)
                if(is_inf):
                    log_warn(f"[Warning] INF encountered in parameter {param_name}. Keep Loading...", on=verbose)
                matched_state_dict[param_name] = param_tensor  
            elif(is_nan):
                e = f"NAN encountered for parameter {param_name}"
                if(strict_check):
                    log_fatal(e, "Quit Job...")
                else:
                    log_warn(e, "Skipping loading...", on=verbose)
            else:  
                e = f"Shape mismatch for parameter {param_name}; Model: {model_param_shape}; Load: {param_tensor.shape}"  
                if(strict_check):
                    log_fatal(e, "Quit Job...")
                else:
                    if(param_tensor.ndim == len(model_param_shape)):
                        log_warn(e, "Skipping loading...", on=verbose)
                    else:
                        minimal_shape = []
                        for ns,nt in zip(param_tensor.shape, model_param_shape):
                            minimal_shape.append(min(ns,nt))
                        minimal_match = model_state_dict[param_name].clone()
                        match_inds = tuple(slice(0, n) for n in minimal_shape)
                        minimal_match[match_inds] = param_tensor[match_inds]
                        matched_state_dict[param_name] = minimal_match
                        log_warn(e, f"Apply fractional loading with shape {minimal_shape}...", on=verbose)
        else:  
            e = f"Parameter name {param_name} not found in the current model"
            if(strict_check):
                raise log_fatal(e, "Quit Job...")
            else:
                log_warn(e, "Skipping loading...", on=verbose)
      
    model.load_state_dict(matched_state_dict, strict=False)  
    return model  

def check_model_validity(model, max_norm_allowed=100.0, verbose=False):
    # Check the validity of model in RunTime
    param_isnormal = dict()
    for param_name, param_tensor in model.named_parameters():
        if(not param_tensor.requires_grad):
            continue # Skip static parameters

        is_nan = torch.isnan(param_tensor).any()
        is_inf = torch.isinf(param_tensor).any()
        l2_norm = torch.norm(param_tensor, p=2).item()
        norm_valid = (l2_norm < max_norm_allowed)

        param_isnormal[param_name] = False
        if(is_nan):
            log_warn(f"NAN encountered in parameter {param_name}.")
        elif(is_inf):
            log_warn(f"INF encountered in parameter {param_name}.")
        elif not norm_valid:
            log_warn(f"Large norm ({l2_norm}) encountered in parameter {param_name}.")
        else:
            param_isnormal[param_name] = True
    return param_isnormal

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
