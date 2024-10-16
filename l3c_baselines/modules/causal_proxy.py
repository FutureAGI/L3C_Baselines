import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from .recursion import PRNN, SimpleLSTM
from .block_wrapper import MultiBlocks
from .transformers import ARTransformerEncoder
from .mamba import MambaBlock
from .blockrec_wrapper import BlockRecurrentWrapper

class CausalBlock(nn.Module):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, config):
        super().__init__()
        self.model_type = config.model_type.lower()

        if(self.model_type == "transformer"):
            main_encoder = ARTransformerEncoder(
                config.num_layers, 
                config.hidden_size, 
                config.nhead, 
                config.position_encoding_size, 
                dim_feedforward=config.inner_hidden_size, 
                dropout=config.dropout, 
                context_window=config.context_window
            )
        elif(self.model_type == "lstm"):
            main_encoder = MultiBlocks(
                SimpleLSTM,
                config.num_layers,
                hidden=config.hidden_size,
                fc_hidden=config.inner_hidden_size,
                fc_dropout=config.dropout,
                io_size=config.hidden_size,
                hidden_size=config.memory_hidden_size,
            )
        elif(self.model_type == "prnn"):
            main_encoder = MultiBlocks(
                PRNN,
                config.num_layers,
                hidden=config.hidden_size,
                fc_hidden=config.inner_hidden_size,
                fc_dropout=config.dropout,
                io_size=config.hidden_size,
                hidden_size=config.memory_hidden_size,
            )
        elif(self.model_type == "mamba"):
            main_encoder = MultiBlocks(
                # This module uses roughly 3 * expand * d_model^2 parameters
                MambaBlock,
                config.num_layers,
                hidden=config.hidden_size,
                fc_hidden=config.inner_hidden_size,
                fc_dropout=config.dropout,
                io_size=config.hidden_size,
                d_state=config.d_state,
                d_conv=config.d_conv,
                layer_idx=0,
                max_position_encoding=config.position_encoding_size,
                expand=config.expand,    # Block expansion factor
            )
        else:
            raise Exception("No such causal model: %s" % config.model_type)
        
        self.need_reset = False
        if(config.use_blockrecurrence):
            main_encoder = BlockRecurrentWrapper(main_encoder, config.memory_length, 
                    memory_type = config.memory_type)
            self.need_reset = True

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
    
    def reset(self):
        if(self.need_reset):
            self.layers.reset()
