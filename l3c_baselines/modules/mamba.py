import torch
from torch import nn
from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams


class MambaCache(object):
    """
    A temporary cache for Mamba
    """
    def __init__(self,
            batch_size:int,
            d_model:int,
            expand:int,
            d_conv:int,
            d_state:int,
            dtype:torch.dtype,
            device:torch.device,
            layer_idx:int=0):
        conv_state = torch.zeros(
            batch_size, d_model * expand, d_conv, device=device, dtype=dtype
        )
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, d_model * expand, d_state, device=device, dtype=dtype
        )
        self.key_value_memory_dict = dict()
        self.key_value_memory_dict[layer_idx] = (conv_state, ssm_state)

        self.bsz = batch_size
        self.d_model = d_model
        self.expand = expand
        self.d_conv = d_conv
        self.d_state = d_state
        self.dtype = dtype
        self.device = device
        self.layer_idx = layer_idx

        self.seqlen_offset = -1

    def clone(self):
        ret = MambaCache(self.bsz, self.d_model, self.expand,
                self.d_conv, self.d_state, self.dtype, self.device, self.layer_idx)
        ret.key_value_memory_dict[self.layer_idx][0].copy_(self.key_value_memory_dict[self.layer_idx][0])
        ret.key_value_memory_dict[self.layer_idx][1].copy_(self.key_value_memory_dict[self.layer_idx][1])
        return ret

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
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.layer_idx = layer_idx
        self.max_position_encoding = max_position_encoding
        self.encoder = Mamba(
                  io_size,
                  d_state=d_state,
                  d_conv=d_conv,
                  expand=expand,
                  layer_idx=self.layer_idx)
        
    def forward(self, x, cache=None, need_cache=False):
        if(need_cache):
            if cache is None:
                cache = MambaCache(x.shape[0], self.hidden_size, self.expand,
                                      self.d_conv, self.d_state, x.dtype, x.device, self.layer_idx)
        else:  # only when cache == None and need_cache==False can we neglect inference_params
            cache=None
        out = self.encoder(x, inference_params=cache)

        if(need_cache):
            return out, cache
        else:
            return out, None
