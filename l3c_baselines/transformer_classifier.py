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
from utils import detached_memory, autoregressive_mask
from transformer_block import TransformerEncoderLayer, TransformerEncoder
from embeddings import UnifiedEmbeddings

class TransformerClassifier(nn.Layer):
    """LSTM Classifier"""

    def __init__(self, d_model, emb_size, nlayers, nhead, normalize_before=True, dropout=0.1, weight_attr=None, bias_attr=None):
        super(TransformerClassifier, self).__init__()

        if not normalize_before:
            self.input_norm = nn.LayerNorm(d_model)
            self.output_norm = None
        else:
            self.input_norm = None
            self.output_norm = nn.LayerNorm(d_model)

        self.embeddings = UnifiedEmbeddings(
            hidden_size=d_model,
            vocab_size=emb_size,
            dropout=dropout,
            use_type=False,
            use_role=False,
            use_turn=False,
            norm=self.input_norm,
            pe_style="rel",
            weight_attr=weight_attr,
        )

        layers = nn.LayerList()
        for i in range(nlayers):
            layers.append(TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                attn_dropout=dropout,
                normalize_before=normalize_before,
                pe_style="rel",
                weight_attr=weight_attr,
                output_weight_attr=weight_attr
            ))
        self.encoder = TransformerEncoder(
            layers,
            dropout=dropout,
            norm=self.output_norm,
            use_recompute=False)

        self.output_mapping = Linear(
            d_model, emb_size,
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
        # [Batch_size, Segment, Embedding_Size]
        emb_out, attn_bias = self.embeddings(
            token_ids=features,
            pos_ids=features,
            type_ids=None,
            role_ids=None,
            turn_ids=None,
            input_mask=autoregressive_mask(features),
            aux_emb=None)

        pe_out = self.embeddings.get_pe_out(
            emb_out,
            mems=mems)

        out = self.encoder(
            emb_out,
            attn_bias,
            pe_out=pe_out,
            same_length=False,
            mems=mems,
            need_all_outputs=True,
        )

        memory = out[1][:-1]
        output = self.output_mapping(out)

        return output, memory
