import copy
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

# Modified from facebookresearch/llama/model.py
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    q0_pos: int = 0,
    k0_pos: int = 0):
    """
    q0_pos: the start position of xq
    k0_pos: the start position of xk
    """
    bsz, q_len, _, _ = xq.shape
    k_len = xk.shape[1]

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xqm_ = freqs_cis[:, q0_pos:(q0_pos + q_len)]
    xkm_ = freqs_cis[:, k0_pos:(k0_pos + k_len)]
    print(xqm_.shape, xq_.shape)
    xq_out = torch.view_as_real(xq_ * xqm_).flatten(3)
    xk_out = torch.view_as_real(xk_ * xkm_).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# Taken from facebookresearch/llama/model.py
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    # Modify to dimension: [1, seq, 1, d_head]
    return freqs_cis.unsqueeze(0).unsqueeze(2)


class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model, nheads, max_steps, dropout):
        super().__init__()

        self.drop_p = dropout
        self.n_heads = nheads
        self.d_head = d_model // nheads
        self.max_seq_len = max_steps

        # Attention
        self.q = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )
        self.k = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )
        self.v = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )
        self.att_proj_linear = nn.Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.resid_dropout = nn.Dropout(dropout)
        self.freqs_cis = precompute_freqs_cis(self.d_head, max_steps)
        print(self.freqs_cis.shape)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        """
        x_shape: [bs, seq, hidden]
        """
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q(x), self.k(x), self.v(x)

        # Reshape for rotary embeddings
        xq = xq.view(batch_size, seq_len, self.n_heads, self.d_head)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.d_head)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.d_head)
        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis)

        # Reshape for attention calculation: (b_sz, n_head, s_len, d_head)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Required as we are not using a nn.Dropout layer
        if self.training:
            att_dropout = self.drop_p
        else:
            att_dropout = 0.0

        # Using beta torch functionality (subject to change)
        # See - https://shorturl.at/jtI17
        att = scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            attn_mask=attn_mask,
            dropout_p=att_dropout,
            is_causal=False,
        )

        # Shape (b_sz, s_len, n_head, d_head)
        out = att.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.n_heads * self.d_head)

        return self.resid_dropout(self.att_proj_linear(out))

if __name__=="__main__":
    rmha = RoPEMultiheadAttention(128, 8, 1024, 0.10)
    inputs = torch.randn(4, 1024, 128)
    attn_mask = torch.randn(1024, 1024)
    output = rmha(inputs, attn_mask)
    print(output.shape)
