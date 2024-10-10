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
        super().__init__()

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
                config.inner_hidden_size
                SimpleLSTM,
                config.num_layers,
                dropout=0.10
            )
        elif(config.model_type == "PRNN"):
            main_encoder = MemoryLayers(
                config.hidden_size,
                config.hidden_size,
                config.inner_hidden_size,
                PRNN,
                config.num_layers,
                dropout=0.10
            )
        elif(config.model_type == "MAMBA"):
            main_encoder = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                config.hidden_size, # Model dimension d_model
                config.num_layers,
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            )
        else:
            raise Exception("No such causal model: %s" % model_type)
        
        if(config.use_layer_norm):
            self.encoder = nn.Sequential(
                main_encoder,
                nn.LayerNorm(config.hidden_size, eps=1.0e-5),
            )
        else:
            self.encoder = main_encoder

        if(config.use_blockrecurrence):
            self.encoder = BlockRecurrentWrapper(self.encoder, config.memory_length)

        self.norm = nn.LayerNorm(self.hidden_size, eps=1.0e-5)
        self.checkpoints_density = config.checkpoints_density

if __name__=='__main__':
    DT = CausalModeling(config)