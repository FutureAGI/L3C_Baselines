import torch
from torch import nn
from fla.models.rwkv6.modeling_rwkv6 import RWKV6Block
from fla.models.rwkv6.configuration_rwkv6 import RWKV6Config
from fla.models.utils import Cache
from l3c_baselines.utils import format_cache, memory_cpy


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

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = [param.new_zeros(batch_size, self.hidden_size),
                 param.new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim)]
        return state
        
    def forward(self, x, cache=None, need_cache=False):
        if(need_cache and cache is None):
            cache = self.encoder.init_state(batch_size=x.shape[0])
            cache = Cache.from_legacy_cache([cache])
    
        use_cache = (cache is not None)
        out, _, new_cache = self.encoder(hidden_states=x, past_key_values=cache, use_cache=use_cache)

        new_cache.states = memory_cpy(new_cache.states)

        return out, new_cache
