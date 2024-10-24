import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from l3c_baselines.modules import MLPEncoder, ResidualMLPDecoder, CausalBlock
from l3c_baselines.utils import format_cache

class SADecisionModel(nn.Module):
    """
    Take Observations and actions; output next state and action.
    """
    def __init__(self, config):
        super().__init__()

        self.causal_model = CausalBlock(config.causal_block)

        # 创建Type向量[1, 1, NP, C]
        type_embeddings = torch.randn(1, 1, 2, config.causal_block.hidden_size)
        self.type_query = nn.Parameter(type_embeddings, requires_grad=True)
        mask_embeddings = torch.randn(1, 1, config.state_encode.input_size)
        self.mask_query = nn.Parameter(mask_embeddings, requires_grad=True)

        self.s_encoder = MLPEncoder(config.state_encode)
        self.a_encoder = MLPEncoder(config.action_encode)
        self.s_decoder = ResidualMLPDecoder(config.state_decode)
        self.a_decoder = ResidualMLPDecoder(config.action_decode)

    def forward(self, s_arr, a_arr, cache=None, need_cache=True, state_dropout=0.0, T=1.0, update_memory=True):
        """
        Input Size:
            observations:[B, NT, H], float
            actions:[B, NT, H], float
            cache: [B, NC, H]
        """
        B = s_arr.shape[0]
        NT = s_arr.shape[1]
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
        outputs, new_cache = self.causal_model(outputs, cache=cache, need_cache=need_cache, update_memory=update_memory)

        # Acqure Outputs: [a_0, s_1, a_1, ...]
        outputs = outputs.reshape(B, NT, 2, -1)

        # Predict s_1, s_2, ..., s_{t+1}
        obs_output = self.s_decoder(outputs[:, :, 1])

        # Predict a_0, a_1, ..., a_t
        act_output = self.a_decoder(outputs[:, :, 0], T=T)

        return obs_output, act_output, new_cache

    def reset(self):
        self.causal_model.reset()


class RSADecisionModel(nn.Module):
    """
    Take Observations, actions and reward; output next state, action and reward.
    """
    def __init__(self, config):
        super().__init__()

        self.causal_model = CausalBlock(config.causal_block)
        self.hidden_size = config.causal_block.hidden_size

        # 创建Type向量[1, 1, NP, C]
        type_embeddings = torch.randn(1, 1, 3, self.hidden_size)
        self.type_query = nn.Parameter(type_embeddings, requires_grad=True)
        mask_embeddings_s = torch.randn(1, 1, self.hidden_size)
        mask_embeddings_r = torch.randn(1, 1, self.hidden_size)
        self.mask_query_s = nn.Parameter(mask_embeddings_s, requires_grad=True)
        self.mask_query_r = nn.Parameter(mask_embeddings_r, requires_grad=True)

        self.s_encoder = MLPEncoder(config.state_encode)
        self.r_encoder = MLPEncoder(config.reward_encode)
        self.a_encoder = MLPEncoder(config.action_encode)
        self.s_decoder = ResidualMLPDecoder(config.state_decode)
        self.a_decoder = ResidualMLPDecoder(config.action_decode)
        self.r_decoder = ResidualMLPDecoder(config.reward_decode)

    def forward(self, s_arr, a_arr, r_arr, cache=None, need_cache=True, state_dropout=0.0, reward_dropout=0.0, T=1.0, update_memory=True):
        """
        Input Size:
            observations:[B, NT, H], float
            actions:[B, NT, H], float
            rewards:[B, NT, H], float
            cache: [B, NC, H]
        """
        B = s_arr.shape[0]
        NT = s_arr.shape[1]
        Ba = a_arr.shape[0]
        NTa = a_arr.shape[1]
        Br = r_arr.shape[0]
        NTr = r_arr.shape[1]

        assert Br == Ba and Ba == B and NTr == NTa and NT == NTr

        # Add state dropouts
        device = s_arr.device
        p_noise = (0.5 * state_dropout * torch.rand((B, 1, 1)) * torch.ones(B, NT, 1)).to(device)
        p_mask = (0.5 * state_dropout * torch.rand((B, 1, 1)) * torch.ones(B, NT, 1)).to(device)
        eps = torch.randn((B, NT, self.hidden_size)).to(device)
        dp_eps = torch.bernoulli(p_noise)
        dp_mask = torch.bernoulli(p_mask)

        # Calculate dropout for mazes: 50% * state_dropout add noise, 50% * state_dropout are directly masked
        observation_in = self.s_encoder(s_arr) + eps * dp_eps
        observation_in = observation_in * (1 - dp_mask) + self.mask_query_s * dp_mask
        observation_in = observation_in.view(B, NT, 1, -1)

        # Add reward dropouts
        pr_noise = (0.5 * reward_dropout * torch.rand((B, 1, 1)) * torch.ones(B, NT, 1)).to(device)
        pr_mask = (0.5 * reward_dropout * torch.rand((B, 1, 1)) * torch.ones(B, NT, 1)).to(device)
        eps = torch.randn((B, NT, self.hidden_size)).to(device)
        dpr_eps = torch.bernoulli(pr_noise)
        dpr_mask = torch.bernoulli(pr_mask)

        # Calculate dropout for mazes: 50% * state_dropout add noise, 50% * state_dropout are directly masked
        reward_in = self.r_encoder(r_arr.view(Br, NTr, 1)) + eps * dpr_eps
        reward_in = reward_in * (1 - dpr_mask) + self.mask_query_r * dpr_mask
        reward_in = reward_in.view(B, NT, 1, -1)

        # Input actions: [B, NT, 1, H]
        action_in = self.a_encoder(a_arr).view(B, NT, 1, -1)

        # [B, NT, 3, H]
        outputs = torch.cat([reward_in, observation_in, action_in], dim=2)

        # Add Type Embedding
        outputs = outputs + self.type_query

        # Concatenate [r_0, s_0, a_0, r_1, s_1, a_1, r_2, ...] to acquire the size of [B, NT * 3, H]
        outputs = outputs.view(B, NT * 3, -1)

        # Temporal Encoders
        outputs, new_cache = self.causal_model(outputs, cache=cache, need_cache=need_cache, update_memory=update_memory)

        # Acqure Outputs: [r0_, a_0, s_1, r_1, ...]
        outputs = outputs.reshape(B, NT, 3, -1)

        # Predict s_1, s_2, ..., s_{t+1}
        obs_output = self.s_decoder(outputs[:, :, 2])

        # Predict a_0, a_1, ..., a_t
        act_output = self.a_decoder(outputs[:, :, 1], T=T)

        # Predict r_0, r_1, ..., r_t
        rew_output = self.r_decoder(outputs[:, :, 2])

        return obs_output, act_output, rew_output, new_cache

    def reset(self):
        self.causal_model.reset()

if __name__=='__main__':
    import sys
    from l3c_baselines.utils import Configure
    config = Configure()
    config.from_yaml(sys.argv[1])
    #SADM = SADecisionModel(config.model_config.decision_block)
    RSADM = SADecisionModel(config.model_config.decision_block)
    cache = None
    for seg in range(10):
        i_s = torch.rand((4, 64, 128))
        i_a = torch.randint(0, 4, (4, 64))
        i_r = torch.rand((4, 64, 1))
        o_r, o_s, o_a, cache = RSADM.forward(i_r, i_s, i_a, cache=cache, update_memory=False)
        print(seg, o_s.shape, o_a.shape, o_r.shape)
        print(format_cache(cache, "Cache"))
        print(format_cache(RSADM.causal_model.layers.memory, "Memory"))
