import torch
from torch import nn
from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams


class MambaBlock(nn.Module):
    def __init__(self,
                io_size: int=512,
                d_state: int=16,
                d_conv: int=4,
                max_position_encoding: int=1024,
                layer_idx: int=0,
                expand: int=2):
        super().__init__()
        self.hidden_size = io_size
        self.layer_idx = layer_idx
        self.encoder = Mamba(d_state=d_state,
                  d_conv=d_conv,
                  expand=expand,
                  layer_idx=self.layer_idx)
        
    def forward(self, x, cache=None, need_cache=False):
        if(need_cache):
            if cache is None:
                kv_cache = dict()
            elif cache is not None:
                kv_cache = {self.layer_idx: (cache[0], cache[1])}
            inference_params=InferenceParams(key_value_memory_dict=dict())
        else:
            inference_params = None
        out = self.encoder(x, inference_params=inference_params)

        if(need_cache):
            return out, inference_params.key_value_memory_dict[self.layer_idx]
        else:
            return out, None
