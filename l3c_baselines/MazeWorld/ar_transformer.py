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
        print(attn_mask)
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
