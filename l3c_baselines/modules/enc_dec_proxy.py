import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from .res_nets import ImageDecoder, ImageEncoder, MLPEncoder, ResidualMLPDecoder
from .proxy_base import ProxyBase


class EncodeBlock(ProxyBase):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, config):
        super().__init__(config)

        if(config.model_type == "ResNet"):
            self.layers = ImageEncoder(
                config.img_size,
                3,
                config.hidden_size,
                config.n_res_block
            )
        elif(config.model_type == "MLP"):
            self.layers = MLPEncoder(
                config.input_type,
                config.input_size,
                config.hidden_size,
                config.dropout,
            )
        else:
            raise Exception("No such causal model: %s" % model_type)

        if(config.has_attr("input_size")):
            self.input_size = config.input_size
        if(isinstance(config.hidden_size, list) or isinstance(config.hidden_size, tuple)):
            self.output_size = config.hidden_size[-1]
        else:
            self.output_size = config.hidden_size
    
class DecodeBlock(ProxyBase):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, config):
        super().__init__(config)

        if(config.model_type == "ResNet"):
            self.layers = ImageDecoder(
                config.img_size,
                config.input_size,
                config.hidden_size,
                3,
                config.n_res_block
            )
        elif(config.model_type == "MLP"):
            self.layers = ResidualMLPDecoder(
                config.output_type,
                config.input_size,
                config.hidden_size,
                dropout = config.dropout,
                layer_norm = config.layer_norm,
                residual_connect = config.residual_connect
            )
        else:
            raise Exception("No such causal model: %s" % model_type)

        if(config.has_attr("input_size")):
            self.input_size = config.input_size
        if(isinstance(config.hidden_size, list) or isinstance(config.hidden_size, tuple)):
            self.output_size = config.hidden_size[-1]
        else:
            self.output_size = config.hidden_size
