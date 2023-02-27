#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformer block."""

import collections

import paddle
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_parallel import ColumnParallelLinear, RowParallelLinear
from paddle.distributed.fleet.recompute import recompute
from paddle.fluid import layers
import paddle.incubate as incubate
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.layer.transformer import _convert_attention_mask


def parallel_matmul(lm_output, logit_weights, parallel_output):
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()
    rank = hcg.get_model_parallel_rank()

    if world_size > 1:
        input_parallel = paddle.distributed.collective._c_identity(
            lm_output, group=model_parallel_group)

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

        if parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(
            logits, group=model_parallel_group)
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        return logits


class MultiHeadAttention(nn.Layer):
    """Multi-Head Attention.

    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.
    """

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 pe_style="abs",
                 weight_attr=None,
                 output_weight_attr=None,
                 bias_attr=None,
                 fuse_qkv=False,
                 num_partitions=1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.pe_style = pe_style

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        assert self.num_heads % num_partitions == 0
        self.num_heads = self.num_heads // num_partitions

        # other
        self.fuse_qkv = fuse_qkv

        self._dtype = self._helper.get_default_dtype()

        if self.fuse_qkv:
            assert self.kdim == embed_dim, "embed_dim should be equal to kdim"
            assert self.vdim == embed_dim, "embed_dim should be equal to vidm"

            self.qkv_proj = ColumnParallelLinear(
                embed_dim,
                3 * embed_dim,
                weight_attr=weight_attr,
                has_bias=bias_attr is not False,
                gather_output=False)
        else:
            self.q_proj = ColumnParallelLinear(
                embed_dim,
                embed_dim,
                weight_attr=weight_attr,
                has_bias=bias_attr is not False,
                gather_output=False)

            self.k_proj = ColumnParallelLinear(
                self.kdim,
                embed_dim,
                weight_attr=weight_attr,
                has_bias=bias_attr is not False,
                gather_output=False)

            self.v_proj = ColumnParallelLinear(
                self.vdim,
                embed_dim,
                weight_attr=weight_attr,
                has_bias=bias_attr is not False,
                gather_output=False)

        if self.pe_style == "rel":
            self.u_bias = self.create_parameter(
                [self.num_heads, self.head_dim],
                attr=weight_attr,
                dtype=self._dtype)
            self.v_bias = self.create_parameter(
                [self.num_heads, self.head_dim],
                attr=weight_attr,
                dtype=self._dtype)
            self.r_proj = nn.Linear(
                self.kdim,
                embed_dim,
                weight_attr=weight_attr,
                bias_attr=bias_attr)

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            weight_attr=output_weight_attr,
            has_bias=bias_attr is not False,
            input_is_parallel=True)

        self.dropout = nn.Dropout(dropout, mode="upscale_in_train")

    def _fuse_prepare_qkv(self, x, mem=None, cache=None):
        """Prapares linear projected queries, keys and values in fused style."""
        mix_layer = self.qkv_proj(x)
        mix_layer = paddle.reshape_(mix_layer, [0, 0, self.num_heads, 3 * self.head_dim])
        mix_layer = paddle.transpose(mix_layer, [0, 2, 1, 3])
        q, k, v = paddle.split(mix_layer, num_or_sections=3, axis=-1)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = paddle.concat([cache.k, k], axis=2)
            v = paddle.concat([cache.v, v], axis=2)
            new_cache = self.Cache(k, v)

        if isinstance(mem, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k = paddle.concat([mem.k, k], axis=2)
            v = paddle.concat([mem.v, v], axis=2)

        if cache is not None:
            return q, k, v, new_cache
        else:
            return q, k, v

    def _prepare_qkv(self, query, key, value, mem=None, cache=None):
        """Prapares linear projected queries, keys and values."""
        q = self.q_proj(query)
        q = paddle.reshape(q, shape=[0, 0, self.num_heads, self.head_dim])
        q = paddle.transpose(q, perm=[0, 2, 1, 3])

        k, v = self._compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = paddle.concat([cache.k, k], axis=2)
            v = paddle.concat([cache.v, v], axis=2)
            new_cache = self.Cache(k, v)

        if isinstance(mem, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k = paddle.concat([mem.k, k], axis=2)
            v = paddle.concat([mem.v, v], axis=2)

        if cache is not None:
            return q, k, v, new_cache
        else:
            return q, k, v

    def _compute_kv(self, key, value):
        """Prapares linear projected  keys and values.

        Applies linear projection on input keys and values, then splits heads
        (reshape and transpose) to get keys and values from different representation
        subspaces. The results are used as key-values pairs for subsequent multiple
        parallel attention.

        It is part of calculations in multi-head attention, and is provided as
        a method to pre-compute and prefetch these results, thus we can use them
        to construct cache for inference.

        """
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = paddle.reshape(k, shape=[0, 0, self.num_heads, self.head_dim])
        k = paddle.transpose(k, perm=[0, 2, 1, 3])
        v = paddle.reshape(v, shape=[0, 0, self.num_heads, self.head_dim])
        v = paddle.transpose(v, perm=[0, 2, 1, 3])
        return k, v

    def _rel_shift(self, x):
        """Apply relative shift to attention score.

        Args:
            x: attention socre, shape: [batch_size, num_heads, qlen, klen]

        Retruns:
            x: processed attention score, shape: [batch_size, num_heads, qlen, klen]
        """
        shape = paddle.shape(x)
        zero_pad = paddle.zeros((shape[0], shape[1], shape[2], 1), dtype=x.dtype) # shape: [B, N, Q, 1]
        x_padded = paddle.concat([zero_pad, x], axis=3) # shape: [B, N, Q, K+1]
        x_padded = paddle.reshape(x_padded, (shape[0], shape[1], shape[3] + 1, shape[2])) # shape: [Q, N, K+1, Q]
        x_padded = x_padded[:, :, 1:] # shape: [Q, N, K, Q]
        x = paddle.reshape(x_padded, shape) # shape: [Q, N, Q, K]
        return x

    def gen_cache(self, key, value=None, type=Cache):
        """Generates cache for faster decoding step by step.

        The generated cache is an instance of `MultiHeadAttention.Cache` or an
        instance of `MultiHeadAttention.StaticCache`.
        """
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v = self._compute_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            k = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            v = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def forward(self,
                query,
                key,
                value,
                attn_bias=None,
                pe_out=None,
                mem=None,
                cache=None,
                need_weights=False):
        key = query if key is None else key
        value = query if value is None else value

        # compute q ,k ,v
        if not self.fuse_qkv and isinstance(mem, paddle.Tensor):
            mlen = paddle.shape(mem)[1]
            query = query[:, mlen:]
        if self.fuse_qkv:
            # TODO: query, key and value has different shape.
            out = self._fuse_prepare_qkv(query, mem, cache)
        else:
            out = self._prepare_qkv(query, key, value, mem, cache)
        if cache is not None:
            q, k, v, new_cache = out
        else:
            q, k, v = out
        if self.fuse_qkv and isinstance(mem, paddle.Tensor):
            mlen = paddle.shape(mem)[1]
            q = q[:, :, mlen:]

        scale_factor = self.head_dim ** -0.5

        if self.pe_style == "rel":
            # attention score: A + C
            q_ac = q + paddle.unsqueeze(self.u_bias, axis=[0, 2])
            product_ac = paddle.einsum("bnqd,bnkd->bnqk", q_ac, k)

            # attention score: B + D
            q_bd = q + paddle.unsqueeze(self.v_bias, axis=[0, 2])
            r = self.r_proj(pe_out) # shape: [K, H]
            r = paddle.reshape(r, [-1, self.num_heads, self.head_dim]) # shape: [K, N, D]
            product_bd = paddle.einsum("bnqd,knd->bnqk", q_bd, r)
            product_bd = self._rel_shift(product_bd)
            # NOTE: fix bidirectional relative position encoding
            product_bd = product_bd[:, :, :, :paddle.shape(product_ac)[-1]]

            # final attention score
            product = product_ac + product_bd
            product *= scale_factor
        else:
            product = layers.matmul(
                x=q, y=k, transpose_y=True, alpha=scale_factor)

        if isinstance(attn_bias, str) and attn_bias == "upper_triangle":
            weights = incubate.softmax_mask_fuse_upper_triangle(product)
        elif attn_bias is not None:
            weights = F.softmax(product + attn_bias)
        else:
            weights = F.softmax(product)
        weights = self.dropout(weights)

        out = paddle.matmul(weights, v)

        # combine heads
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(new_cache)
        return out if len(outs) == 1 else tuple(outs)


class TransformerEncoderLayer(nn.Layer):
    """Transformer encoder layer.

    It contains Multi-head Attention and Position-wise Feed-forward Network.
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="gelu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=True,
                 pe_style="abs",
                 weight_attr=None,
                 output_weight_attr=None,
                 bias_attr=None,
                 fuse_qkv=False,
                 num_partitions=1):
        super(TransformerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            pe_style=pe_style,
            weight_attr=weight_attr,
            output_weight_attr=output_weight_attr,
            bias_attr=bias_attr,
            fuse_qkv=fuse_qkv,
            num_partitions=num_partitions)

        self.linear1 = ColumnParallelLinear(
            d_model,
            dim_feedforward,
            weight_attr=weight_attr,
            gather_output=False,
            has_bias=bias_attr is not False)

        self.linear2 = RowParallelLinear(
            dim_feedforward,
            d_model,
            weight_attr=output_weight_attr,
            input_is_parallel=True,
            has_bias=bias_attr is not False)

        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self,
                src,
                attn_bias=None,
                pe_out=None,
                mem=None,
                cache=None):
        residual = src

        if isinstance(mem, paddle.Tensor):
            cat = paddle.concat([mem, src], axis=1)
        else:
            cat = src

        if self.normalize_before:
            cat = self.norm1(cat)

        if cache is not None:
            src, incremental_cache = self.self_attn(cat, cat, cat, attn_bias, pe_out, mem=mem, cache=cache)
        else:
            src = self.self_attn(cat, cat, cat, attn_bias, pe_out, mem=mem)

        src = residual + self.dropout1(src)

        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)

        # src = self.linear2(F.gelu(self.linear1(src), approximate=True))
        src = self.activation(self.linear1(src))
        src = self.dropout2(src)
        src = self.linear2(src)

        src = residual + self.dropout1(src)

        if not self.normalize_before:
            src = self.norm2(src)

        if cache is not None:
            return src, incremental_cache
        else:
            return src

    def gen_cache(self, x):
        """Generates cache for faster decoding step by step.

        The generated cache is Cache or StaticCache produced by `MultiHeadAttention.gen_cache`.
        See `MultiHeadAttention.gen_cache` for more details.
        """
        incremental_cache = self.self_attn.gen_cache(x, type=self.self_attn.Cache)
        return incremental_cache


class TransformerEncoder(nn.Layer):
    """TransformerEncoder is a stack of N encoder layers."""

    def __init__(self, encoder_layers, dropout, norm=None, use_recompute=False):
        super(TransformerEncoder, self).__init__()
        # TODO: use LayerList (https://github.com/PaddlePaddle/Paddle/blob/bed652d6ece3791c6a68d0a61f0f1007fc044a91/python/paddle/nn/layer/transformer.py#L652)
        self.layers = encoder_layers
        self.norm = norm
        self.use_recompute = use_recompute
        self.dropout = nn.Dropout(dropout, mode="upscale_in_train")

    def forward(self,
                src,
                src_mask=None,
                pe_out=None,
                same_length=False,
                mems=None,
                caches=None,
                need_all_outputs=False):
        attn_bias = _convert_attention_mask(src_mask, src.dtype)

        output = src
        if caches is not None:
            new_caches = []

        if mems is not None:
            if isinstance(mems, (list, tuple)):
                mem = mems[0]
            else:
                mem = mems
            if attn_bias is not None:
                src_shape = paddle.shape(src)
                bsz = src_shape[0]
                slen = src_shape[1]
                if isinstance(mem, paddle.Tensor):
                    mlen = paddle.shape(mem)[1]
                elif isinstance(mem, MultiHeadAttention.StaticCache):
                    mlen = paddle.shape(mem)[2]
                mem_attn_mask = paddle.ones([slen, mlen], dtype=attn_bias.dtype)
                if same_length:
                    mem_attn_mask = paddle.triu(mem_attn_mask)
                mem_attn_bias = paddle.scale(mem_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
                mem_attn_bias = paddle.expand(mem_attn_bias, [bsz, 1, slen, mlen])
                attn_bias = paddle.concat([mem_attn_bias, attn_bias], axis=3)
        else:
            mem = None

        all_outputs = [output]
        for i, mod in enumerate(self.layers):
            # NOTE: Support different memory each layer or same memory each layer
            if isinstance(mems, (list, tuple)):
                mem = mems[i]
            else:
                mem = mems
            if caches is not None:
                output, new_cache = mod(output, attn_bias, pe_out, mem=mem, cache=caches[i])
                new_caches.append(new_cache)
            elif self.use_recompute and self.training:
                output = recompute(mod, output, attn_bias, pe_out, mem)
            else:
                output = mod(output, attn_bias, pe_out, mem=mem)
            all_outputs.append(output)
        if self.use_recompute and self.training:
            self.checkpoints = [out.name for out in all_outputs]

        if self.norm is not None:
            output = self.norm(output)

        output = self.dropout(output)

        outs = [output]
        if need_all_outputs:
            outs.append(all_outputs)
        if caches is not None:
            outs.append(new_caches)
        return outs[0] if len(outs) == 1 else tuple(outs)

    def update_mems(self, mems, reset_flag, new_mems, max_mlen):
        """Update memory.

        Concat old memories and new memories, only stay the latest memories and detach all of them.

        Args:
            mems ([`Tensor`] or List of [`Tensor`] or List of [`MultiHeadAttention.StaticCache`]):
                old memories, the shape of each Tensor is [batch_size, memory_len, hidden_size] or
                [batch_size, num_heads, memory_len, head_dim].
            reset_flag ([`Tensor`]): whether to reset memory.

        Returns:
            updated_mems ([`Tensor`] or List of [`Tensor`] or List of [`MultiHeadAttention.StaticCache`]):
                updated memories, the shape of each Tensor is [batch_size, memory_len, hidden_size] or
                [batch_size, num_heads, memory_len, head_dim].
        """
        if isinstance(new_mems, paddle.Tensor): # shape: [B, M1, H], [B, M2, H]
            if mems is None:
                updated_mems = new_mems
            else:
                updated_mems = paddle.concat([mems, new_mems], axis=1)
            updated_mems = updated_mems[:, -max_mlen:] * (1 - reset_flag)
            return updated_mems.detach()
        elif isinstance(new_mems, (list, tuple)):
            if len(new_mems) == 0:
                raise ValueError("No memory in memories.")
            elif isinstance(new_mems[0], paddle.Tensor):
                reset_flag = paddle.unsqueeze(reset_flag, axis=[1, 2]) # shape: [B, 1, 1]
            elif isinstance(new_mems[0], MultiHeadAttention.StaticCache):
                reset_flag = paddle.unsqueeze(reset_flag, axis=[1, 2, 3]) # shape: [B, 1, 1, 1]
            else:
                raise ValueError(f"Cannot recognize memory type: {type(new_mems[0])}")
            if mems is None:
                mems = [None] * len(new_mems)
        else:
            raise ValueError(f"Cannot recognize memories type: {type(mems)}")

        updated_mems = []
        if len(mems) != len(new_mems):
            raise ValueError(f"The size of old memories and new memories is not match: {len(mems)} vs {len(new_mems)}")
        for mem, new_mem in zip(mems, new_mems):
            if isinstance(new_mem, paddle.Tensor): # shape: [B, M1, H], [B, M2, H]
                if mem is None:
                    updated_mem = new_mem
                else:
                    updated_mem = paddle.concat([mem, new_mem], axis=1)
                updated_mem = updated_mem[:, -max_mlen:] * (1 - reset_flag)
                updated_mems.append(updated_mem.detach()) # shape: [B, N, M1, D], [B, N, M2, D]
            elif isinstance(new_mem, MultiHeadAttention.StaticCache):
                if mem is None:
                    updated_k = new_mem.k
                    updated_v = new_mem.v
                else:
                    updated_k = paddle.concat([mem.k, new_mem.k], axis=2)
                    updated_v = paddle.concat([mem.v, new_mem.v], axis=2)
                updated_k = updated_k[:, :, -max_mlen:] * (1 - reset_flag)
                updated_v = updated_v[:, :, -max_mlen:] * (1 - reset_flag)
                updated_mems.append(
                    MultiHeadAttention.StaticCache(updated_k.detach(), updated_v.detach()))
        return updated_mems

    def gen_cache(self, x, do_zip=False):
        """Generates cache for faster decoding step by step.

        The generated cache is a list, and each element in it is Cache or StaticCache
        produced by `TransformerLayer.gen_cache`. See `TransformerLayer.gen_cache`
        for more details. If `do_zip` is True, apply `zip` on these tuples to get
        a list with two elements.
       """
        cache = [layer.gen_cache(x) for layer in self.layers]
        if do_zip:
            cache = list(zip(*cache))
        return cache
