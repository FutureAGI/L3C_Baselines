import copy
import torch
import torch.nn as nn
from .rope_mha import RoPEMultiheadAttention, precompute_freqs_cis
from torch.utils.checkpoint import checkpoint

class ARTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(ARTransformerEncoderLayer, self).__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout)

        # Define other layers (e.g., Feedforward, LayerNorm, Dropout) here
        # Norm First = True
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1.0e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1.0e-5)
        self.activation = nn.GELU()

    def forward(self, 
                src : torch.Tensor, 
                rope : torch.Tensor, 
                attn_mask : torch.Tensor, 
                cache=None):
        """
        Cache: B, NT, H
        SRC: Other Parts
        """
        # Self Attention
        if cache is not None:
            q0_pos=cache.shape[1]
            kv = self.norm1(torch.cat([cache, src], dim=1))
            output = kv[:, q0_pos:]
        else:
            q0_pos=0
            output = self.norm1(src)
            kv = output
        
        output = self.self_attn(output, kv, kv, rope, attn_mask=attn_mask, q0_pos=q0_pos)

        # Residual Connection
        output = src + output

        # FeedForward + Residual
        output = output + self.dropout(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(output))))))

        # Apply other layers and return output
        return output

class ARTransformerEncoder(nn.Module):
    def __init__(self, 
            num_layers : int, 
            d_model : int, 
            nhead : int, 
            max_steps : int,
            dim_feedforward : int=2048, 
            dropout : float=0.10,
            context_free: bool=False):
        super(ARTransformerEncoder, self).__init__()
        ar_layer = ARTransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.layers = nn.ModuleList([copy.deepcopy(ar_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.d_head = d_model // nhead
        self.max_steps = max_steps

        attn_mask = (torch.triu(torch.ones(max_steps, max_steps)) == 1).transpose(1, 0)
        attn_mask = attn_mask.float().masked_fill(attn_mask == False, float('-inf')).masked_fill(attn_mask == True, float(0.0))

        # If context-free, only window of 2 is allowed
        if(context_free):
            ext_mask = (torch.triu(torch.ones(max_steps, max_steps), diagonal=-1) == 1)
            attn_mask = attn_mask.masked_fill(ext_mask == False, float('-inf'))
            print("[Warning] Context-Free Model, Use an attention mask of {attn_mask}")

        self.rope_embedding = precompute_freqs_cis(self.d_head, self.max_steps)
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, src, cache=None, need_cache=False, checkpoints_density=-1):
        # Every checkpoints_density we arrange a checkpoint
        # If checkpoints_density < 1 we do not use checkpoints
        # Calculate Cache Size
        l = src.shape[1]
        if(cache is None):
            s = l
        else:
            s = l + cache[0].shape[1]
            
        new_cache = None

        output=src
        if(need_cache):
            if(cache is not None):
                new_cache = [torch.cat([cache[0], output.detach()], dim=1)]
            else:
                new_cache = [output.detach()]
        for i, layer in enumerate(self.layers):
            if(checkpoints_density < 1):
                need_checkpoint=False
            elif((i + 1) % checkpoints_density == 0):
                need_checkpoint=True
            else:
                need_checkpoint=False
            if(cache is not None):
                if(not need_checkpoint):
                    output = layer(output, self.rope_embedding, self.attn_mask[-l:, -s:], cache[i]) 
                else:
                    output = checkpoint(lambda x: layer(x, self.rope_embedding, self.attn_mask[-l:, -s:], cache[i]), output)
                if(need_cache):
                    new_cache.append(torch.cat([cache[i + 1], output.detach()], dim=1))
            else:
                if(not need_checkpoint):
                    output = layer(output, self.rope_embedding, self.attn_mask[-l:, -s:])
                else:
                    output = checkpoint(lambda x: layer(x, self.rope_embedding, self.attn_mask[-l:, -s:]), output)
                if(need_cache):
                    new_cache.append(output.detach())
        return output, new_cache

class DecisionTransformer(nn.Module):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, 
            observation_size, 
            action_vocab_size, 
            num_layers, 
            d_model, 
            nhead, 
            max_time_step, 
            dropout=0.1, 
            checkpoints_density=-1,
            context_free=False):
        super().__init__()

        self.d_model = d_model
        self.max_time_step = max_time_step
        self.num_layers = num_layers
        self.checkpoints_density=checkpoints_density

        # 创建动作编码层
        self.action_vocab_size = action_vocab_size
        self.action_embedding = nn.Embedding(action_vocab_size, d_model)

        # 创建Transformer编码器层
        self.pre_layer = nn.Linear(observation_size, d_model)
        max_seq_len = 2 * max_time_step + 1
        self.encoder = ARTransformerEncoder(num_layers, d_model, nhead, max_seq_len, dim_feedforward=4*d_model, dropout=dropout, context_free=context_free)

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
            actions:[B, NT], int 
            cache: [B, NC, H]
        """
        B, NT, H = observations.shape
        assert actions.shape[0] == B and actions.shape[1] == NT, "The shape of actions should be [%s, %s], but get %s" % (B, NT, actions.shape)

        #calculate cached positions
        if(cache is None):
            cache_len = 0
        else:
            assert isinstance(cache, list) and len(cache) == self.num_layers + 1, "The cache must be list with length == num_layers + 1"
            cache_len = cache[0].shape[1]

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
        action_in = self.action_embedding(actions).view(B, NT, 1, -1)

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

class ARTransformerStandard(nn.Module):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, vocab_size, num_layers, d_model, nhead, max_time_step, dropout=0.10, checkpoints_density=-1):
        super().__init__()

        self.d_model = d_model
        self.max_time_step = max_time_step
        self.num_layers = num_layers
        self.checkpoints_density=checkpoints_density

        # 创建动作编码层
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(vocab_size, d_model)

        # 创建Transformer编码器层
        self.encoder = ARTransformerEncoder(num_layers, d_model, nhead, max_time_step, dim_feedforward=4*d_model, dropout=dropout)
        self.norm = nn.LayerNorm(d_model, eps=1.0e-5)

        self.output_mapping = nn.Sequential(nn.Linear(d_model, vocab_size), nn.Softmax(dim=-1))

    def forward(self, inputs, cache=None, need_cache=True):
        """
        Input Size:
            inputs:[B, NT], int
        """
        B, NT = inputs.shape

        #calculate cached positions
        if(cache is None):
            cache_len = 0
        else:
            assert isinstance(cache, list) and len(cache) == self.num_layers + 1, "The cache must be list with length == num_layers + 1"
            cache_len = cache[0].shape[1]

        # Input actions: [B, NT, 1, H]
        outputs = self.word_embedding(inputs)

        outputs, new_cache = self.encoder(outputs, cache=cache, need_cache=need_cache, checkpoints_density=self.checkpoints_density)

        outputs = self.output_mapping(self.norm(outputs))

        return outputs, new_cache

if __name__=='__main__':
    DT = DecisionTransformer(256, 5, 2, 64, 8, 64, dropout=0.0, checkpoints_density=3)
    inputs_obs = torch.randn((1, 64, 256))
    input_acts = torch.randint(0, 4, (1, 64))
    out_obs_1, out_act_1, cache_1 = DT(inputs_obs[:, :32], input_acts[:, :32], need_cache=True)
    out_obs_2, out_act_2, cache_2 = DT(inputs_obs[:, 32:], input_acts[:, 32:], cache=cache_1, need_cache=True)
    out_obs_3, out_act_3, cache_3 = DT(inputs_obs, input_acts, need_cache=True)
    print(out_obs_3[:, :32] - out_obs_1)
    print(out_act_3[:, :32] - out_act_1)
    print(out_obs_3[:, 32:] - out_obs_2)
    print(out_act_3[:, 32:] - out_act_2)
    print(cache_3[0][:, :64] - cache_1[0])
    print(cache_3[0] - cache_2[0])
    print(cache_3[1][:, :64] - cache_1[1])
    print(cache_3[1] - cache_2[1])


    inputs2 = torch.randint(0, 1024, (4, 64))
    ART2 = ARTransformerStandard(1024, 8, 128, 8, 1024, checkpoints_density=4)
    out_nlp, cache = ART2(inputs2, need_cache=True)
    out_nlp2, cache = ART2(inputs2, cache=cache, need_cache=True)
    print(out_nlp.shape, out_nlp2.shape)
