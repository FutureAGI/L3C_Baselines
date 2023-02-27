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
import sys
import paddle
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_parallel import ColumnParallelLinear, RowParallelLinear
from paddle.distributed.fleet.recompute import recompute
from paddle.fluid import layers
from paddle.nn import Linear
import paddle.incubate as incubate
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.layer.transformer import _convert_attention_mask
from plastic_layers import FullPlasticLayer
from transformer_block import MultiHeadAttention
from transformer_block import TransformerEncoderLayer

class PlasmerEncoderLayer(nn.Layer):
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
                 fuse_qkv=False,
                 weight_attr=None,
                 bias_attr=None,
                 num_partitions=1):
        super(PlasmerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.weight_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0e-3, name=None),
                need_clip=True)
        self.bias_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(mean=-1.0, std=0.0, name=None),
                need_clip=True)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            pe_style=pe_style,
            fuse_qkv=fuse_qkv,
            num_partitions=num_partitions,
            weight_attr=weight_attr
            )

        self.plastic_fc = FullPlasticLayer(
                d_model,
                d_model,
                plasticity_rule="mABCD",
                dopamine_activation="sigmoid",
                memory_decay=0.01,
                memory_decay_thres=5.0,
                initial_variance=0.10,
                learning_rate=0.01,
                rules_attr=self.weight_attr,
                dopamine_weight_attr=self.weight_attr,
                dopamine_bias_attr=self.bias_attr)

        self.linear1 = ColumnParallelLinear(
            d_model,
            dim_feedforward,
            weight_attr=weight_attr,
            gather_output=False,
            has_bias=bias_attr is not False)

        self.linear2 = RowParallelLinear(
            dim_feedforward,
            d_model,
            weight_attr=weight_attr,
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

        if self.normalize_before:
            tgt = self.norm1(src)

        if cache is not None:
            tgt, incremental_cache = self.self_attn(src, src, src, attn_bias, pe_out, cache=cache)
        else:
            tgt = self.self_attn(src, src, src, attn_bias, pe_out)

        if(mem is None):
            source_shape = src.shape
            mem = paddle.normal(
                    mean=0.0, 
                    std=self.plastic_fc.initial_variance, 
                    shape=(source_shape[0], source_shape[2], source_shape[2]),
                    name=None)

        # Residual After the plastic Layer
        tgt = self.plastic_fc(src, mem) + self.dropout1(tgt)
        updated_mem = self.plastic_fc.update_mem(src, tgt, mem)

        src = residual + tgt

        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)

        src = self.activation(self.linear1(src))
        src = self.dropout2(src)
        src = self.linear2(src)

        src = residual + self.dropout1(src)

        if not self.normalize_before:
            src = self.norm2(src)

        if cache is not None:
            return (src, updated_mem), incremental_cache
        else:
            return (src, updated_mem)

    def gen_cache(self, x):
        """Generates cache for faster decoding step by step.

        The generated cache is Cache or StaticCache produced by `MultiHeadAttention.gen_cache`.
        See `MultiHeadAttention.gen_cache` for more details.
        """
        incremental_cache = self.self_attn.gen_cache(x, type=self.self_attn.Cache)
        return incremental_cache
