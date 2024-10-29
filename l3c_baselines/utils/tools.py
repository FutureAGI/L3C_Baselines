import os
import re
import sys
import yaml
import torch
import numpy
from torch import nn
import torch.distributed as dist
from types import SimpleNamespace
from copy import deepcopy
from dateutil.parser import parse
from collections import defaultdict

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

def custom_load_model(model, state_dict_path, black_list=[], max_norm_allowed=1.0e+2, strict_check=False):  
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
                    print(f"Parameter hits {param_name} black lists {name}")
                    hits_black = True
            if(hits_black):
                continue

            if model_param_shape == param_tensor.shape and (not is_nan):  
                if(not norm_valid):
                    print(f"[Warning] Large norm ({l2_norm}) encountered in parameter {param_name}. Keep Loading...")
                if(is_inf):
                    print(f"[Warning] INF encountered in parameter {param_name}. Keep Loading...")
                matched_state_dict[param_name] = param_tensor  
            elif(is_nan):
                e = f"NAN encountered for parameter {param_name}"
                if(strict_check):
                    raise Exception(e, "Quit Job...")
                else:
                    print(e, "Skipping loading...")
            else:  
                e = f"Shape mismatch for parameter {param_name}; Model: {model_param_shape}; Load: {param_tensor.shape}"  
                if(strict_check):
                    raise Exception(e, "Quit Job...")
                else:
                    print(e, "Skipping loading...")
        else:  
            e = f"Parameter name {param_name} not found in the current model"
            if(strict_check):
                raise Exception(e, "Quit Job...")
            else:
                print(e, "Skipping loading...")
      
    model.load_state_dict(matched_state_dict, strict=False)  
    return model  

def check_model_validity(model, max_norm_allowed=100.0):
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
            print(f"[Warning] NAN encountered in parameter {param_name}.")
        elif(is_inf):
            print(f"[Warning] INF encountered in parameter {param_name}.")
        elif not norm_valid:
            print(f"[Warning] Large norm ({l2_norm}) encountered in parameter {param_name}.")
        else:
            param_isnormal[param_name] = True
    return param_isnormal

def img_pro(observations):
    return observations / 255

def img_post(observations):
    return observations * 255

def print_memory(info="Default"):
    print(info, "Memory allocated:", torch.cuda.memory_allocated(), "Memory cached:", torch.cuda.memory_cached())

def gradient_failsafe(model, optimizer, scaler):
    overflow=False
    for param in model.parameters():
        if param.grad is not None and (torch.isinf(param.grad).any() or torch.isnan(param.grad).any()):
            print("Warning: Gradient contains inf or nan, setting those gradients to zero.")
            overflow=True
    if(overflow):
        for param in model.parameters():
            param.grad.zero_()
        scaler.unscale_(optimizer)
        optimizer.__setstate__({'state': defaultdict(dict)})

class DistStatistics(object):
    """
    Provide distributed statistics
    """
    def __init__(self, *keys, verbose=False):
        self.keys = keys
        if("count" in keys):
            self.is_average = True
            if(verbose):
                log_debug("Found 'count' keyword in keys, statistics will be averaged by count")
        else:
            self.is_average = False
            if(verbose):
                log_debug("No 'count' keyword detected, statistics will be summed but not averaged")
        self.reset()

    def reset(self):
        self._data = dict()
        for key in self.keys:
            self._data[key] = []

    def add_with_safety(self, device, **kwargs):
        zeroflag = False
        if("count" in kwargs):
            cnt = kwargs["count"]
        else:
            cnt = 1
        for key, value in kwargs.items():
            if torch.isinf(value).any() or torch.isnan(value).any():
                print(f"[WARNING] 'Device:{device}' stating '{key}' suffering prediction loss = NAN/INF, fill with 0")
                zeroflag = True
        safe_stats = dict()
        if zeroflag:
            for key, value in kwargs.items():
                safe_stats[key] = torch.zeros_like(value)
        else:
            for key, value in kwargs.items():
                safe_stats[key] = value.clone().detach()
        for key, value in safe_stats.items():
            if(key not in self._data):
                raise KeyError(f"Key {key} not registered in Statistics class")
            if(key != "count"):
                value *= cnt
            dist.all_reduce(value.data)
            self._data[key].append(value.cpu().detach())

    def __call__(self):
        stat_res = dict()
        for key in self.keys:
            stat_res[key] = torch.stack(self._data[key]).sum(dim=0)
            if(len(stat_res[key].shape) < 1 or stat_res[key].numel() < 2):
                stat_res[key] = float(stat_res[key])
        if(self.is_average):
            for key in self.keys:
                if(key != "count"):
                    stat_res[key] /= float(stat_res["count"])

        return stat_res

def rewards2go(rewards, gamma=0.98):
    """
    returns a future moving average of rewards
    """
    rolled_rewards = rewards.clone()
    r2go = rewards
    n = max(min(50, -1/numpy.log10(gamma)), 0)
    for _ in range(n):
        rolled_rewards = gamma * torch.roll(rolled_rewards, shifts=-1, dims=1)
        r2go += rolled_rewards
    return r2go
