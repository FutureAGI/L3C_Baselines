import torch
from torch import nn
from fla.layers.gsa import GatedSlotAttention
from fla.layers.gla import GatedLinearAttention
from fla.models.utils import Cache


class GSABlock(nn.Module):
    def __init__(self,
                io_size: int=512,
                num_heads: int=4,
                num_slots: int=4,
                layer_idx: int=0):
        super().__init__()
        self.hidden_size = io_size
        self.layer_idx = layer_idx
        self.encoder = GatedSlotAttention(
                  hidden_size=io_size,
                  num_heads=num_heads,
                  num_slots=num_slots,
                  layer_idx=0)
        
    def forward(self, x, cache=None, need_cache=False):
        if(need_cache and cache is None):
            cache = self.encoder.init_state(x.shape[0])
            cache = Cache.from_legacy_cache([cache])
    
        use_cache = (cache is not None)
        out, _, new_cache = self.encoder(hidden_states=x, past_key_values=cache, use_cache=use_cache)

        if(new_cache is not None and need_cache):
            for state in new_cache.states:
                for sta in state:
                    sta.detach_()

        return out, new_cache
        

class GLABlock(nn.Module):
    def __init__(self,
                io_size: int=512,
                num_heads: int=4,
                layer_idx: int=0):
        super().__init__()
        self.hidden_size = io_size
        self.layer_idx = layer_idx
        self.encoder = GatedLinearAttention(
                  hidden_size=io_size,
                  num_heads=num_heads,
                  layer_idx=0)
        
    def forward(self, x, cache=None, need_cache=False):
        if(need_cache and cache is None):
            cache = self.encoder.init_state(x.shape[0])
            cache = Cache.from_legacy_cache([cache])
    
        use_cache = (cache is not None)
        out, _, new_cache = self.encoder(hidden_states=x, past_key_values=cache, use_cache=use_cache)

        if(new_cache is not None and need_cache):
            for state in new_cache.states:
                for sta in state:
                    sta.detach_()

        return out, new_cache
