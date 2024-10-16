import torch
from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams


class MambaBlocks(nn.Module):
    def __init__(self, num_layers: int, 
                hidden_size: int,
                d_state: int,
                d_conv: int,
                max_position_encoding: int,
                layer_idx: int,
                expand: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer_idx = layer_idx
        self.encoder = Mamba(d_state=d_state,
                  d_conv=d_conv,
                  expand=expand,
                  layer_idx=i) for i in range(num_layers)
        
    def forward(self, x, cache=None, need_cache=False):
        if need_cache:
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
        else：
            return out, None

