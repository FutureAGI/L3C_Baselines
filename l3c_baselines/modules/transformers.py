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
            context_window: int=-1):
        super(ARTransformerEncoder, self).__init__()
        ar_layer = ARTransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.layers = nn.ModuleList([copy.deepcopy(ar_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.d_head = d_model // nhead
        self.max_steps = max_steps

        attn_mask = (torch.triu(torch.ones(max_steps, max_steps)) == 1).transpose(1, 0)
        attn_mask = attn_mask.float().masked_fill(attn_mask == False, float('-inf')).masked_fill(attn_mask == True, float(0.0))

        # If context-free, only window of 2 is allowed
        if(context_window > -1):
            ext_mask = (torch.triu(torch.ones(max_steps, max_steps), diagonal=-context_window) == 1)
            attn_mask = attn_mask.masked_fill(ext_mask == False, float('-inf'))
            print(f"[Warning] Context-Window is applied, Use an attention mask of {attn_mask}")

        self.rope_embedding = precompute_freqs_cis(self.d_head, self.max_steps)
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, src, cache=None, need_cache=False, checkpoints_density=-1):
        # Every checkpoints_density we arrange a checkpoint
        # If checkpoints_density < 1 we do not use checkpoints
        # Calculate Cache Size
        l = src.shape[1]
        ks = 0
        if(cache is None):
            qs = 0
            e = l
        else:
            qs = cache[0].shape[1]
            e = qs + l
            
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
                    output = layer(output, self.rope_embedding, self.attn_mask[qs:e, ks:e], cache[i]) 
                else:
                    output = checkpoint(lambda x: layer(x, self.rope_embedding, self.attn_mask[qs:e, ks:e], cache[i]), output)
                if(need_cache):
                    new_cache.append(torch.cat([cache[i + 1], output.detach()], dim=1))
            else:
                if(not need_checkpoint):
                    output = layer(output, self.rope_embedding, self.attn_mask[qs:e, ks:e])
                else:
                    output = checkpoint(lambda x: layer(x, self.rope_embedding, self.attn_mask[qs:e, ks:e]), output)
                if(need_cache):
                    new_cache.append(output.detach())
        return output, new_cache

class ARTransformerStandard(nn.Module):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, vocab_size, num_layers, d_model, nhead, max_time_step, dropout=0.10, checkpoints_density=-1, d_model_inner=None):
        super().__init__()

        self.d_model = d_model
        self.max_time_step = max_time_step
        self.num_layers = num_layers
        self.checkpoints_density=checkpoints_density

        # 创建动作编码层
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(vocab_size, d_model)

        # 创建Transformer编码器层
        if(d_model_inner is None):
            d_model_inner = 4 * d_model
        self.encoder = ARTransformerEncoder(num_layers, d_model, nhead, max_time_step, dim_feedforward=d_model_inner, dropout=dropout)
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
    inputs = torch.randint(0, 1024, (4, 64))
    ART2 = ARTransformerStandard(1024, 8, 128, 8, 1024, checkpoints_density=4)
    out_nlp, cache = ART2(inputs, need_cache=True)
    out_nlp2, cache = ART2(inputs, cache=cache, need_cache=True)
    print(out_nlp.shape, out_nlp2.shape)
