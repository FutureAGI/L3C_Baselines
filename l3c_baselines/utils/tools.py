import os
import sys
import torch
from torch import nn
import yaml
from types import SimpleNamespace
from copy import deepcopy
from dateutil.parser import parse
from collections import defaultdict
import re


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

def infer_type(s):
    s = s.strip().lower()

    # If starts with specific tokens return string
    if(s.startswith("\"") or s.startswith("\'")):
        return s.strip("\"").strip("\'")

    # Check for boolean
    if s in ['true', 'false']:
        return s == 'true'
    elif s in ['yes', 'no']:
        return s == 'yes'
    
    # None type
    if s in ['None', 'null', 'none', 'NONE', 'NULL', 'Null']:
        return None
    
    # Check for integer
    try:
        return int(s)
    except ValueError:
        pass

    # Check for float
    try:
        return float(s)
    except ValueError:
        pass

    # Check for date/time
    try:
        return parse(s)
    except ValueError:
        pass

    # Check for list (e.g., "[1, 2, 3]" or "[1,2,3]")
    if re.match(r'^\[.*\]$', s):
        try:
            return eval(s)
        except:
            pass

    # Default to string
    return s

class Configure(object):
    def __init__(self, data=None):
        super().__setattr__("__config", dict())
        if(data is not None):
            super().__getattribute__("__config").update(data)

    def from_yaml(self, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        super().__getattribute__("__config").update(data)

    def from_dict(self, data):
        super().__getattribute__("__config").update(data)

    def clear(self):
        config = super().__getattribute__("__config")
        for key in config:
            del config[key]

    def __getattr__(self, attr):
        config = super().__getattribute__("__config")
        if attr in config:
            value = config[attr]
            if isinstance(value, dict):
                return Configure(value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        d = super().__getattribute__("__config")
        if('.' in attr):
            keys = attr.split('.')
            for key in keys[:-1]:
                if(key not in d):
                    d[key] = dict()
                d = d[key]
            d[keys[-1]] = infer_type(value)
        else:
            d[attr] = infer_type(value)

    def set_value(self, attr, value):
        self.__setattr__(attr, value)

    def __repr__(self):
        config = super().__getattribute__("__config")
        def rec_prt(d, n_tab):
            _repr=""
            for k in d:
                v = d[k]
                if(isinstance(v, dict)):
                    _repr += "\t"*n_tab
                    _repr += f"{k}:\n"
                    _repr += rec_prt(v, n_tab + 1)
                else:
                    _repr += "\t"*n_tab
                    _repr += f"{k} = {v}\n"
            return _repr
        return f"\n\n{self.__class__.__name__}\n\n" + rec_prt(config, 0) + "\n\n"
