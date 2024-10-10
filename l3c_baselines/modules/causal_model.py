import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from l3c_baselines.modules import Mamba, PRNN, SimpleLSTM, MemoryLayers, Mamba, ARTransformerEncoder, BlockWrapper


class CausalModel(nn.Module):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.inner_hidden_size = config.inner_hidden_size
        self.position_encoding_size = config.position_encoding_size
        self.num_layers = config.num_layers
        self.nhead = config.nhead
        self.dropout = config.dropout
        self.checkpoints_density = config.checkpoints_density
        self.use_block_wrapper = config.use_block_wrapper
        self.model_type = config.model_type
        self.context_window = config.context_window
        self.memory_length = config.memory_length

        if(self.model_type == "TRANSFORMER"):
            self.encoder = ARTransformerEncoder(
                config.num_layers, 
                config.hidden_size, 
                config.nhead, 
                config.position_encoding_size, 
                dim_feedforward=config.inner_hidden_size, 
                dropout=config.dropout, 
                context_window=config.context_window
            )
        elif(model_type == "LSTM"):
            self.encoder = MemoryLayers(
                config.hidden_size,
                config.hidden_size,
                config.inner_hidden_size
                SimpleLSTM,
                config.num_layers,
                dropout=0.10
            )
        elif(model_type == "PRNN"):
            self.encoder = MemoryLayers(
                config.hidden_size,
                config.hidden_size,
                config.inner_hidden_size,
                PRNN,
                config.num_layers,
                dropout=0.10
            )
        elif(model_type == "MAMBA"):
            self.encoder = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                config.hidden_size, # Model dimension d_model
                config.num_layers,
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            )
        else:
            raise Exception("No such causal model: %s" % model_type)

        if(config.use_block_wrapper):
            self.encoder = BlockReccurrentWrapper(self.encoder, config.memory_length)

        self.checkpoints_density = config.checkpoints_density

    def forward(self, *args, **kwargs):
        output, new_cache = self.encoder(*args, **kwargs)

        return output, new_cache

if __name__=='__main__':
    DT = CausalModeling(config)