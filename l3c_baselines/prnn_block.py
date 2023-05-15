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
from .plastic_layers import FullPlasticLayer
from .transformer_block import MultiHeadAttention
from .transformer_block import TransformerEncoderLayer

class PRNNEncoderLayer(nn.Layer):
    """Transformer encoder layer.

    It contains Multi-head Attention and Position-wise Feed-forward Network.
    """

    def __init__(self, d_model, activation="tanh"):
        super(PRNNEncoderLayer, self).__init__()

        self.weight_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0e-3, name=None),
                need_clip=True)
        self.bias_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(mean=-1.0, std=0.0, name=None),
                need_clip=True)

        self.plastic_hh = FullPlasticLayer(
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

        self.plastic_ih = FullPlasticLayer(
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

        self.activation = getattr(F, activation)

    def forward(self, src, mem=None):
        if(mem is not None):
			mem_ih, mem_hh, mem_h = mem
		else:
            mem_ih, mem_hh, mem_h = (None, None, None)

        # Residual After the plastic Layer
        tgt_1, updated_mem_ih = self.plastic_ih(src, mem_ih)
        tgt_2, updated_mem_hh = self.plastic_hh(mem_h, mem_hh)

		tgt = self.activation(tgt_1 + tgt_2)

        return tgt, (updated_mem_ih, updated_mem_hh, tgt)
