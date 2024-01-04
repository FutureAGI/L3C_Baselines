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
import numpy as np

import paddle
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_parallel import ColumnParallelLinear, RowParallelLinear
from paddle.fluid import layers
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.layer.transformer import TransformerEncoderLayer, TransformerEncoder


class AutoRegressiveTRN(nn.Layer):
    """RMT Encoder Model"""

    def __init__(self,
            vocab_size,
            nhidden,
            nlayer, 
            nhead, 
            seg_len=256,
            weight_attr=None, 
            bias_attr=None):
        super(AutoRegressiveTRN, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, nhidden)
        self.pos_embeddings = nn.Embedding(seg_len, nhidden)

        encoder_layer = TransformerEncoderLayer(nhidden, nhead, nhidden*4)
        self.encoder = TransformerEncoder(encoder_layer, nlayer)
        self.n_layer = nlayer
        self.n_hidden = nhidden

        self.output_mapping = nn.Linear(
            nhidden, vocab_size,
            weight_attr,
            bias_attr)

        self.mask = paddle.tensor.triu(
            paddle.full(
                shape=[seg_len, seg_len],
                fill_value=-np.inf,
                dtype=paddle.get_default_dtype(),
            ),
            1,
        )

    def forward(self, inputs, mems=None):
        """
        src: features, labels - [Batch_size, Segment], Int
        """
        # [Batch_size, Segment, Embedding_Size]
        bsz, seg = inputs.shape

        type_inputs = paddle.zeros((bsz, seg), dtype=paddle.get_default_dtype())
        token_embs = self.word_embeddings(inputs)
        pos_embs = paddle.unsqueeze(self.pos_embeddings.weight[:seg], axis=0)

        features = token_embs + pos_embs

        out = self.encoder(
            features,
            self.mask[:seg, :seg]
        )

        outputs = self.output_mapping(out)

        return outputs, None

if __name__=="__main__":
    vocab = 200
    model = AutoRegressiveTRN(vocab, 128, 2, 8, seg_len=256)
    input_token = paddle.randint(low=0, high=vocab, shape=[64, 256], dtype="int32")
    outputs, mems = model.forward(input_token)
    print("1st output", outputs.shape)
    outputs, mems = model.forward(input_token, mems=mems)
    print("2nd output", outputs.shape)
