import copy
import torch
import torch.nn as nn

class ARTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(ARTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Define other layers (e.g., Feedforward, LayerNorm, Dropout) here
        # Norm First = True
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1.0e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1.0e-5)
        self.activation = nn.GELU()

    def forward(self, src, attn_mask, cache=None):
        """
        Cache: B, NT, H
        SRC: Other Parts
        """

        # Norm first always
        output = self.norm1(src)

        # Self Attention
        if cache is not None:
            kv = torch.cat([src, cache], dim=1)
            output, _ = self.self_attn(output, kv, kv, attn_mask=attn_mask)
        else:
            output, _ = self.self_attn(output, output, output, attn_mask=attn_mask)
        
        # Residual Connection
        output = src + output

        # FeedForward + Residual
        output = output + self.dropout(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(output))))))

        # Apply other layers and return output
        return output

class ARTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(ARTransformerEncoder, self).__init__()
        ar_layer = ARTransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.layers = nn.ModuleList([copy.deepcopy(ar_layer) for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, cache=None, need_cache=False):
        # Calculate Cache Size
        l = src.shape[1]
        if(cache is None):
            s = l
        else:
            s = l + cache[0].shape[1]
            
        attn_mask = (torch.triu(torch.ones(s, s)) == 1).transpose(1, 0)[-l:]
        attn_mask = attn_mask.float().masked_fill(attn_mask == False, float('-inf')).masked_fill(attn_mask == True, float(0.0))
        attn_mask = attn_mask.to(src.device)
        new_cache = None
        output=src
        if(need_cache):
            if(cache is not None):
                new_cache = [torch.cat([cache[0], output.detach()], dim=1)]
            else:
                new_cache = [output.detach()]
        for i, layer in enumerate(self.layers):
            if(cache is not None):
                output = layer(output, attn_mask, cache[i]) 
                if(need_cache):
                    new_cache.append(torch.cat([cache[0], output.detach()], dim=1))
            else:
                output = layer(output, attn_mask)
                if(need_cache):
                    new_cache.append(output.detach())
        return output, new_cache


class DecisionTransformer(nn.Module):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, observation_size, action_vocab_size, num_layers, d_model, nhead, max_time_step):
        super().__init__()

        self.d_model = d_model
        self.max_time_step = max_time_step
        self.num_layers = num_layers

        # 创建动作编码层
        self.action_vocab_size = action_vocab_size
        self.action_embedding = nn.Embedding(action_vocab_size, d_model)

        # 创建Transformer编码器层
        self.pre_layer = nn.Linear(observation_size, d_model)
        self.encoder = ARTransformerEncoder(num_layers, d_model, nhead, dim_feedforward=4*d_model)

        # 创建位置编码和Query向量[1, NT, 1, C]
        temporal_embeddings = torch.randn(1, self.max_time_step + 1, 1, d_model)
        self.temporal_query = nn.Parameter(temporal_embeddings, requires_grad=True)

        # 创建Type向量[1, 1, NP, C]
        type_embeddings = torch.randn(1, 1, 2, d_model)
        self.type_query = nn.Parameter(type_embeddings, requires_grad=True)


    def forward(self, observations, actions, cache=None, need_cache=True):
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

        # Input actions: [B, NT, 1, H]
        action_in = self.action_embedding(actions).view(B, NT, 1, -1)
        observation_in = self.pre_layer(observations).view(B, NT, 1, -1)

        # [B, NT, 2, H]
        outputs = torch.cat([observation_in, action_in], dim=2)

        # Add Temporal Position Embedding
        outputs = outputs + self.temporal_query[:, cache_len:(NT + cache_len)]

        # Add Type Embedding
        outputs = outputs + self.type_query

        # Concatenate [s_0, a_0, s_1, a_1, s_2, ...] to acquire the size of [B, NT * 2, H]
        outputs = outputs.view(B, NT * 2, -1)

        # Temporal Encoders
        outputs, new_cache = self.encoder(outputs, cache=cache, need_cache=need_cache)

        # Acqure Outputs: [a_0, s_1, a_1, ...]
        outputs = outputs.reshape(B, NT, 2, -1)

        # Predict a_0, a_1, ..., a_t
        act_output = outputs[:, :, 0]
        # Predict s_1, s_2, ..., s_{t+1}
        obs_output = outputs[:, :, 1]

        return obs_output, act_output, new_cache

if __name__=='__main__':
    ART = ARTransformerEncoder(8, 256, 8, 1024, 0.1)

    inputs = torch.randn((4, 64, 256))
    # Test ART without Cache
    output1, cache1 = ART(inputs)
    print(output1.shape, cache1)
    # Test ART adding Cache 
    output2, cache2 = ART(inputs, need_cache=True)
    output3, cache3 = ART(inputs, cache=cache2, need_cache=True)
    output4, cache4 = ART(inputs, cache=cache3, need_cache=True)
    print(output2.shape, len(cache2), cache2[0].shape)
    print(output3.shape, len(cache3), cache3[0].shape)
    print(output4.shape, len(cache4), cache4[0].shape)

    DT = DecisionTransformer(256, 5, 8, 128, 8, 1024)
    input_acts = torch.randint(0, 4, (4, 64))
    out_obs, out_act, cache = DT(inputs, input_acts)
    print(out_obs.shape, out_act.shape, len(cache))
