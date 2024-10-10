import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from l3c_baselines.modules import EncodeBlock, DecodeBlock, CausalBlock
from l3c_baselines.utils import format_cache

class SADecisionModel(nn.Module):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, config):
        super().__init__()

        self.causal_model = CausalBlock(config.causal_block)

        # 创建Type向量[1, 1, NP, C]
        type_embeddings = torch.randn(1, 1, 2, config.causal_block.hidden_size)
        self.type_query = nn.Parameter(type_embeddings, requires_grad=True)
        mask_embeddings = torch.randn(1, 1, config.state_encode.input_size)
        self.mask_query = nn.Parameter(mask_embeddings, requires_grad=True)

        self.s_encoder = EncodeBlock(config.state_encode)
        self.a_encoder = EncodeBlock(config.action_encode)
        self.s_decoder = DecodeBlock(config.state_decode)
        self.a_decoder = DecodeBlock(config.action_decode)

    def forward(self, s_arr, a_arr, cache=None, need_cache=True, state_dropout=0.0, T=1.0):
        """
        Input Size:
            observations:[B, NT, H], float
            actions:[B, NT, H], float
            cache: [B, NC, H]
        """
        B, NT, H = s_arr.shape
        Ba = a_arr.shape[0]
        NTa = a_arr.shape[1]

        assert Ba == B and NTa == NT

        # Add state dropouts
        device = s_arr.device
        p_noise = (0.5 * state_dropout * torch.rand((B, 1, 1)) * torch.ones(B, NT, 1)).to(device)
        p_mask = (0.5 * state_dropout * torch.rand((B, 1, 1)) * torch.ones(B, NT, 1)).to(device)
        eps = torch.randn((B, NT, H)).to(device)
        dp_eps = torch.bernoulli(p_noise)
        dp_mask = torch.bernoulli(p_mask)

        # Calculate dropout for mazes: 50% * state_dropout add noise, 50% * state_dropout are directly masked
        observation_in = s_arr + eps * dp_eps
        observation_in = observation_in * (1 - dp_mask) + self.mask_query * dp_mask
        observation_in = self.s_encoder(observation_in).view(B, NT, 1, -1)

        # Input actions: [B, NT, 1, H]
        action_in = self.a_encoder(a_arr).view(B, NT, 1, -1)

        # [B, NT, 2, H]
        outputs = torch.cat([observation_in, action_in], dim=2)

        # Add Type Embedding
        outputs = outputs + self.type_query

        # Concatenate [s_0, a_0, s_1, a_1, s_2, ...] to acquire the size of [B, NT * 2, H]
        outputs = outputs.view(B, NT * 2, -1)

        # Temporal Encoders
        outputs, new_cache = self.causal_model(outputs, cache=cache, need_cache=need_cache)

        # Acqure Outputs: [a_0, s_1, a_1, ...]
        outputs = outputs.reshape(B, NT, 2, -1)

        # Predict s_1, s_2, ..., s_{t+1}
        obs_output = self.s_decoder(outputs[:, :, 1])

        # Predict a_0, a_1, ..., a_t
        act_output = self.a_decoder(outputs[:, :, 0], T=T)

        return obs_output, act_output, new_cache

    def reset(self):
        self.causal_model.reset()

if __name__=='__main__':
    import sys
    from l3c_baselines.utils import Configure
    config = Configure()
    config.from_yaml(sys.argv[1])
    SADM = SADecisionModel(config.model_config.decision_block)
    cache = None
    for seg in range(10):
        i_s = torch.rand((4, 64, 128))
        i_a = torch.randint(0, 4, (4, 64))
        o_s, o_a, cache = SADM.forward(i_s, i_a, cache=cache)
        print(seg, o_s.shape, o_a.shape)
        print(format_cache(cache, "Cache"))
        print(format_cache(SADM.causal_model.layers.memory, "Memory"))
