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
"""Plasmer Classifier"""

import collections
import numpy
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
from .embeddings import UnifiedEmbeddings
from .plasmer_block import PlasmerEncoderLayer 
from .utils import detached_memory, autoregressive_mask

class PlasmerClassifier(nn.Layer):
    """Plasmer is a stack of N encoder layers."""

    def __init__(self, 
            d_model, 
            nhead, 
            dim_feedforward, 
            nlayers,
            vocab_size,
            dropout=0.1, 
            norm=None, 
            normalize_before=True,
            use_recompute=False, 
            weight_attr=None, 
            bias_attr=None):
        super(PlasmerClassifier, self).__init__()

        self.norm = norm
        self.use_recompute = use_recompute
        self.dropout = nn.Dropout(dropout, mode="upscale_in_train")

        if not normalize_before:
            self.input_norm = nn.LayerNorm(d_model)
            self.output_norm = None
        else:
            self.input_norm = None
            self.output_norm = nn.LayerNorm(d_model)

        self.embeddings = UnifiedEmbeddings(
            hidden_size=d_model,
            vocab_size=vocab_size,
            dropout=dropout,
            use_type=False,
            use_role=False,
            use_turn=False,
            norm=self.input_norm,
            pe_style="rel",
            weight_attr=weight_attr,
        )

        self.plastic_fc = FullPlasticLayer(
            d_model,
            d_model,
            plasticity_rule="mABCD",
            dopamine_activation="tanh",
            memory_decay=0.01,
            memory_decay_thres=5.0,
            initial_variance=0.10,
            learning_rate=0.05,
            rules_attr=weight_attr,
            dopamine_weight_attr=weight_attr,
            dopamine_bias_attr=bias_attr)

        self.layers = nn.LayerList([PlasmerEncoderLayer(
             d_model, 
             nhead, 
             dim_feedforward,
             normalize_before=True,
             plastic_fc=self.plastic_fc,
             ) for _ in range(nlayers)])

        self.output_mapping = Linear(
            d_model, vocab_size,
            weight_attr,
            bias_attr)

    def forward(self,
                features,
                src_mask=None,
                pe_out=None,
                mems=None):
        """
        src: features, labels - [Batch_size, Segment], Int
        """
        attn_bias = _convert_attention_mask(src_mask, paddle.get_default_dtype())

        # [Batch_size, Segment, Embedding_Size]
        emb_out, attn_bias = self.embeddings(
            token_ids=features,
            pos_ids=None,
            type_ids=None,
            role_ids=None,
            turn_ids=None,
            input_mask=autoregressive_mask(features),
            aux_emb=None)

        pe_out = self.embeddings.get_pe_out(
            emb_out,
            mems=mems)

        # Flatten to [Batch_size, Segment, Embedding_Size], with the form of fea, label, fea, label, ...
        if mems is None:
            source_shape = emb_out.shape
            mems = [paddle.normal(
                    mean=0.0, 
                    std=self.layers[0].plastic_fc.initial_variance, 
                    shape=(source_shape[0], source_shape[2], source_shape[2]),
                    name=None) for _ in range(len(self.layers))]

        output = emb_out
        all_outputs = [output]
        new_mems = []
        for i, mod in enumerate(self.layers):
            mem = mems[i]
            if self.use_recompute and self.training:
                output, update_mem = recompute(mod, output, attn_bias, pe_out, mem)
            else:
                output, update_mem = mod(output, attn_bias=attn_bias, pe_out=pe_out, mem=mem)
            new_mems.append(update_mem)
            all_outputs.append(output)

        if self.use_recompute and self.training:
            self.checkpoints = [out.name for out in all_outputs]

        if self.norm is not None:
            output = self.norm(output)

        output = self.output_mapping(self.dropout(output))

        return output, new_mems
