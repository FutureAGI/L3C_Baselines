import torch
from torch import nn
from fla.layers.gsa import GatedSlotAttention
from fla.layers.gla import GatedLinearAttention


class GSABlock(nn.Module):
    def __init__(self,
                io_size: int=512,
                num_heads: int=4,
                num_slots: int=None):
        super().__init__()
        self.hidden_size = io_size
        self.encoder = GatedSlotAttention(
                  hidden_size=io_size,
                  num_heads=num_heads,
                  num_slots=num_slots,
                  layer_idx=0)
        
    def forward(self, x, cache=None, need_cache=False):
        if(need_cache and cache is None): # either cache != None or need_cache == True we must use inference_params
            cache = self.encoder.init_state(x.shape[0])
    
        out, _, new_cache = self.encoder(hidden_states=x, past_key_values=cache, use_cache=(cache is not None))

        if(need_cache):
            return out, new_cache
        else:
            return out, None
        

class GLABlock(nn.Module):
    def __init__(self,
                io_size: int=512,
                num_heads: int=4):
        super().__init__()
        self.hidden_size = io_size
        self.encoder = GatedLinearAttention(
                  hidden_size=io_size,
                  num_heads=num_heads,
                  layer_idx=0)
        
    def forward(self, x, cache=None, need_cache=False):
        if(need_cache and cache is None): # either cache != None or need_cache == True we must use inference_params
            cache = self.encoder.init_state(x.shape[0])
    
        out, _, new_cache = self.encoder(hidden_states=x, past_key_values=cache, use_cache=(cache is not None))

        if(need_cache):
            return out, new_cache
        else:
            return out, None