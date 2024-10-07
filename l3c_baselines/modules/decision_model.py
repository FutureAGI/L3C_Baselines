import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from .mamba_minimal import Mamba
from .recursion import PRNN, SimpleLSTM, MemoryLayers
from .transformers import ARTransformerEncoder


class CausalDecisionModel(nn.Module):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, 
            observation_size,
            num_layers, 
            d_model, 
            nhead, 
            max_time_step, 
            dropout=0.1, 
            context_window=-1,
            inner_hidden_size=None,
            checkpoints_density=-1,
            model_type="LSTM"):
        super().__init__()

        self.d_model = d_model
        self.max_time_step = max_time_step
        self.num_layers = num_layers

        self.checkpoints_density = checkpoints_density

        # 创建Transformer编码器层
        self.pre_layer = nn.Linear(observation_size, d_model)
        max_seq_len = 2 * max_time_step + 1
        if(inner_hidden_size is None):
            inner_hidden_size = 4 * d_model

        if(model_type == "TRANSFORMER"):
            self.encoder = ARTransformerEncoder(
                num_layers, 
                d_model, 
                nhead, 
                max_seq_len, 
                dim_feedforward=inner_hidden_size, 
                dropout=dropout, 
                context_window=context_window
            )
        elif(model_type == "LSTM"):
            self.encoder = MemoryLayers(
                self.d_model,
                self.d_model,
                4 * self.d_model,
                SimpleLSTM,
                num_layers,
                dropout=0.10
            )
        elif(model_type == "PRNN"):
            self.encoder = MemoryLayers(
                self.d_model,
                self.d_model,
                4 * self.d_model,
                PRNN,
                num_layers,
                dropout=0.10
            )
        elif(model_type == "MAMBA"):
            self.encoder = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                self.d_model, # Model dimension d_model
                num_layers,
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            )
        else:
            raise Exception("No such causal model: %s" % model_type)

        self.norm = nn.LayerNorm(d_model, eps=1.0e-5)

        # 创建Type向量[1, 1, NP, C]
        type_embeddings = torch.randn(1, 1, 2, d_model)
        self.type_query = nn.Parameter(type_embeddings, requires_grad=True)
        mask_embeddings = torch.randn(1, 1, observation_size)
        self.mask_query = nn.Parameter(mask_embeddings, requires_grad=True)

    def forward(self, observations, actions, cache=None, need_cache=True, state_dropout=0.0):
        """
        Input Size:
            observations:[B, NT, H], float
            actions:[B, NT, H], float
            cache: [B, NC, H]
        """
        B, NT, H = observations.shape
        assert actions.shape[0] == B and actions.shape[1] == NT and actions.shape[2] == self.d_model, \
                "The shape of actions should be [%s, %s, %s], but get %s" % (B, NT, self.d_model, actions.shape)

        # Add state dropouts
        device = observations.device
        p_noise = (0.5 * state_dropout * torch.rand((B, 1, 1)) * torch.ones(B, NT, 1)).to(device)
        p_mask = (0.5 * state_dropout * torch.rand((B, 1, 1)) * torch.ones(B, NT, 1)).to(device)
        eps = torch.randn((B, NT, H)).to(device)
        dp_eps = torch.bernoulli(p_noise)
        dp_mask = torch.bernoulli(p_mask)

        # Calculate dropout for mazes: 50% * state_dropout add noise, 50% * state_dropout are directly masked
        observation_in = observations + eps * dp_eps
        observation_in = observation_in * (1 - dp_mask) + self.mask_query * dp_mask
        observation_in = self.pre_layer(observation_in).view(B, NT, 1, -1)

        # Input actions: [B, NT, 1, H]
        action_in = actions.view(B, NT, 1, -1)

        # [B, NT, 2, H]
        outputs = torch.cat([observation_in, action_in], dim=2)

        # Add Type Embedding
        outputs = outputs + self.type_query

        # Concatenate [s_0, a_0, s_1, a_1, s_2, ...] to acquire the size of [B, NT * 2, H]
        outputs = outputs.view(B, NT * 2, -1)

        # Temporal Encoders
        outputs, new_cache = self.encoder(outputs, cache=cache, need_cache=need_cache, checkpoints_density=self.checkpoints_density)

        # Acqure Outputs: [a_0, s_1, a_1, ...]
        outputs = self.norm(outputs)
        outputs = outputs.reshape(B, NT, 2, -1)

        # Predict a_0, a_1, ..., a_t
        act_output = outputs[:, :, 0]
        # Predict s_1, s_2, ..., s_{t+1}
        obs_output = outputs[:, :, 1]

        return obs_output, act_output, new_cache

if __name__=='__main__':
    DT = CausalDecisionModel(256, 5, 2, 64, 8, 64, dropout=0.0, checkpoints_density=-1, model_type="LSTM")
    inputs_obs = torch.randn((1, 64, 256))
    input_acts = torch.randint(0, 4, (1, 64, 256))
    out_obs_1, out_act_1, cache_1 = DT(inputs_obs[:, :32], input_acts[:, :32], need_cache=True)
    out_obs_2, out_act_2, cache_2 = DT(inputs_obs[:, 32:], input_acts[:, 32:], cache=cache_1, need_cache=True)
    out_obs_3, out_act_3, cache_3 = DT(inputs_obs, input_acts, need_cache=True)
    print(out_obs_3[:, :32] - out_obs_1)
    print(out_act_3[:, :32] - out_act_1)
    print(out_obs_3[:, 32:] - out_obs_2)
    print(out_act_3[:, 32:] - out_act_2)
