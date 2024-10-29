import torch
from torch import nn
from fla.layers.gsa import GatedSlotAttention
from fla.layers.gla import GatedLinearAttention
from fla.models.utils import Cache


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
            state = self.encoder.init_state(x.shape[0])
            cache = Cache.from_legacy_cache([state])
    
        use_cache = (cache is not None)
        out, _, new_cache = self.encoder(hidden_states=x, past_key_values=state, use_cache=use_cache)

        for state in new_cache.states:
            for sta in state:
                sta.detach_()

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

        kvs = Cache.from_legacy_cache([cache]) 
        out, _, new_cache = self.encoder(hidden_states=x, past_key_values=kvs, use_cache=(cache is not None))

        if(need_cache):
            return out, new_cache[0]
        else:
            return out, None
