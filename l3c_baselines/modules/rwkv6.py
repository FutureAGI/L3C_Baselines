import torch
from torch import nn
from fla.models.rwkv6.modeling_rwkv6 import RWKV6Block
from fla.models.rwkv6.configuration_rwkv6 import RWKV6Config
from fla.models.utils import Cache
from l3c_baselines.utils import format_cache, memory_cpy, log_warn


class RWKV6Layer(nn.Module):
    def __init__(self,
                io_size: int=512,
                expand_k: float = 0.5,
                expand_v: float = 1,
                hidden_ratio: float = 3.5,
                intermediate_size: int = 1024,
                num_heads: int = 4,
                layer_idx: int = 0):
        super().__init__()
        self.config = RWKV6Config(
                  hidden_size=io_size,
                  expand_k=expand_k,
                  expand_v=expand_v,
                  hidden_ratio=hidden_ratio,
                  intermediate_size=intermediate_size,
                  num_heads=num_heads)
        self.layer_idx = layer_idx
        self.encoder = RWKV6Block(
                  self.config,
                  layer_idx=0)

    def init_state(self, batch_size: int):
        param = next(self.parameters())
        state = [param.new_zeros(batch_size, self.encoder.attn.hidden_size),
                 param.new_zeros(batch_size, self.encoder.attn.num_heads, 
                                self.encoder.attn.head_qk_dim, self.encoder.attn.head_v_dim),
                 param.new_zeros(batch_size, self.encoder.ffn.hidden_size)]
        return state

    def forward(self, x, cache=None, need_cache=False):
        if(need_cache and cache is None):
            cache = Cache.from_legacy_cache(None)

        use_cache = (cache is not None)

        out, _, new_cache = self.encoder(hidden_states=x, past_key_values=cache, use_cache=use_cache)

        if(new_cache is not None):
            new_cache.states = memory_cpy(new_cache.states)

        return out, new_cache
