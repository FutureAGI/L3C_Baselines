import os
import sys
import torch
from torch import nn

def show_bar(fraction, bar):
    percentage = int(bar * fraction)
    empty = bar - percentage
    sys.stdout.write("\r") 
    sys.stdout.write("[") 
    sys.stdout.write("=" * percentage)
    sys.stdout.write(" " * empty)
    sys.stdout.write("]") 
    sys.stdout.write("%.2f %%" % (percentage * 100 / bar))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_path(save_model_path, epoch_id):
    directory_path = '%s/%02d/' % (save_model_path, epoch_id)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return (f'{directory_path}/model.pth', f'{directory_path}/vae_optimizer.pth', f'{directory_path}/seq_optimizer.pth') 

def custom_load_model(model, state_dict_path, black_list=dict(), max_norm_allowed=100.0):  
    # 加载保存的模型参数  
    saved_state_dict = torch.load(state_dict_path)  
      
    # 获取当前模型的state_dict  
    model_state_dict = model.state_dict()  
      
    # 初始化一个空的字典，用于存放匹配且形状一致的参数  
    matched_state_dict = {}  
      
    # 遍历保存的模型参数  
    for param_name, param_tensor in saved_state_dict.items():  
        # 检查参数名称是否在当前模型中  
        if param_name in model_state_dict:  
            # 获取当前模型中对应名称的参数形状  
            model_param_shape = model_state_dict[param_name].shape  
              
            # 检查形状是否一致  
            is_nan = torch.isnan(param_tensor).any()
            is_inf = torch.isinf(param_tensor).any()
            l2_norm = torch.norm(param_tensor, p=2).item()
            norm_valid = (l2_norm < max_norm_allowed)

            if param_name in black_list:
                print(f"Skip black list parameter {param_name}")
            elif model_param_shape == param_tensor.shape and (not is_nan):  
                if(not norm_valid):
                    print(f"[Warning] Large norm ({l2_norm}) encountered in parameter {param_name}. Keep Loading...")
                if(is_inf):
                    print(f"[Warning] INF encountered in parameter {param_name}. Keep Loading...")
                matched_state_dict[param_name] = param_tensor  
            elif(is_nan):
                print(f"NAN encountered for parameter {param_name}. Skipping loading...")
            else:  
                print(f"Shape mismatch for parameter {param_name}. Skipping loading...")  
        else:  
            print(f"Parameter name {param_name} not found in the current model. Skipping loading...")  
      
    # 加载匹配的参数到模型中  
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
