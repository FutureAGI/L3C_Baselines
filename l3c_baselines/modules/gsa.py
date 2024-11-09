import torch
from torch import nn
from fla.layers.gsa import GatedSlotAttention
from fla.layers.gla import GatedLinearAttention
from fla.models.utils import Cache
from l3c_baselines.utils import format_cache, memory_cpy        

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
        
    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        state += (param.new_zeros(batch_size, 
                                  self.encoder.num_kv_heads, 
                                  self.encoder.head_k_dim, 
                                  self.encoder.num_slots),
                  param.new_zeros(batch_size, 
                                  self.encoder.num_kv_heads, 
                                  self.encoder.num_slots, 
                                  self.encoder.head_v_dim))
        return state
        
    def forward(self, x, cache=None, need_cache=False):
        if(need_cache and cache is None):
            cache = self.encoder.init_state(x.shape[0])
            cache = Cache.from_legacy_cache([cache])
    
        use_cache = (cache is not None)
        out, _, new_cache = self.encoder(hidden_states=x, past_key_values=cache, use_cache=use_cache)

        if(new_cache is not None and need_cache):
            new_cache.states = memory_cpy(new_cache.states)

        return out, new_cache
    
class GSABlock(GLABlock):
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
        
    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        state += (param.new_zeros(batch_size, 
                                  self.encoder.num_heads, 
                                  self.encoder.head_qk_dim, 
                                  self.encoder.head_v_dim))
        return state