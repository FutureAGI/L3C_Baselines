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
        self.max_position_encoding = max_position_encoding
        self.encoder = Mamba(
                  io_size,
                  d_state=d_state,
                  d_conv=d_conv,
                  expand=expand,
                  layer_idx=self.layer_idx)
        
    def forward(self, x, cache=None, need_cache=False):
        if(need_cache or cache is not None): # either cache != None or need_cache == True we must use inference_params
            if cache is None:
                kv_cache = dict()
            elif cache is not None:
                kv_cache = {self.layer_idx: cache}
            inference_params=InferenceParams(max_seqlen=self.max_position_encoding, 
                                max_batch_size=x.shape[0],
                                key_value_memory_dict=kv_cache)
        else:  # only when cache == None and need_cache==False can we neglect inference_params
            inference_params = None
        out = self.encoder(x, inference_params=inference_params)

        if(need_cache):
            return out, inference_params.key_value_memory_dict[self.layer_idx]
        else:
            return out, None
