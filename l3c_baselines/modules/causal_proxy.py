import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from .mamba_minimal import Mamba
from .recursion import PRNN, SimpleLSTM, MemoryLayers
from .transformers import ARTransformerEncoder
from .blockrec_wrapper import BlockRecurrentWrapper
from .proxy_base import ProxyBase

class CausalBlock(ProxyBase):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, config):
        super().__init__(config)

        if(config.model_type == "TRANSFORMER"):
            main_encoder = ARTransformerEncoder(
                config.num_layers, 
                config.hidden_size, 
                config.nhead, 
                config.position_encoding_size, 
                dim_feedforward=config.inner_hidden_size, 
                dropout=config.dropout, 
                context_window=config.context_window
            )
        elif(config.model_type == "LSTM"):
            main_encoder = MemoryLayers(
                config.hidden_size,
                config.hidden_size,
                config.inner_hidden_size,
                SimpleLSTM,
                config.num_layers,
                dropout=config.dropout
            )
        elif(config.model_type == "PRNN"):
            main_encoder = MemoryLayers(
                config.hidden_size,
                config.hidden_size,
                config.inner_hidden_size,
                PRNN,
                config.num_layers,
                dropout=config.dropout
            )
        elif(config.model_type == "MAMBA"):
            main_encoder = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                config.hidden_size, # Model dimension d_model
                config.num_layers,
                d_state=config.d_state,  # SSM state expansion factor
                d_conv=config.d_conv,    # Local convolution width
                expand=config.expand,    # Block expansion factor
            )
        else:
            raise Exception("No such causal model: %s" % model_type)
        
        if(config.use_blockrecurrence):
            main_encoder = BlockRecurrentWrapper(main_encoder, config.memory_length, 
                    memory_type = config.memory_type)

        if(config.use_layer_norm):
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1.0e-5)
        else:
            self.layer_norm = nn.Identity()

        self.layers = main_encoder
        self.checkpoints_density = config.checkpoints_density

    def forward(self, *args, **kwargs):
        kwargs["checkpoints_density"] = self.checkpoints_density
        out, cache = self.layers.forward(*args, **kwargs)
        return self.layer_norm(out), cache

if __name__=='__main__':
    DT = CausalModeling(config)
