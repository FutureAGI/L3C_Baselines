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
from paddle.nn.layer.rnn import RNNBase, RNN, BiRNN

class bLSTMCell(nn.LSTMCell):
    def __init__(self, scale_factor=10.0, **kwargs):
        super(bLSTMCell, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def forward(self, inputs, states=None):
        if states is None:
            states = self.get_initial_states(inputs, self.state_shape)
        pre_hidden, pre_cell = states
        gates = paddle.matmul(inputs, self.weight_ih, transpose_y=True)
        if self.bias_ih is not None:
            gates = gates + self.bias_ih
        gates += paddle.matmul(pre_hidden, self.weight_hh, transpose_y=True)
        if self.bias_hh is not None:
            gates = gates + self.bias_hh

        chunked_gates = paddle.split(gates, num_or_sections=4, axis=-1)

        i = self._gate_activation(chunked_gates[0])
        f = self._gate_activation(chunked_gates[1])
        o = self._gate_activation(chunked_gates[3])
        c = self.scale_factor * F.tanh(
                1.0/self.scale_factor * (f * pre_cell + i * self._activation(chunked_gates[2])))
        h = o * self._activation(c)

        return h, (h, c)

class LSTMClassifier(nn.Layer):
    """LSTM Classifier"""

    def __init__(self, d_model, emb_size, nlayers=1, dropout=0.10, tied_nlayers=1, weight_attr=None, bias_attr=None):
        super(LSTMClassifier, self).__init__()

        self.fea_embedding = paddle.nn.Embedding(emb_size, d_model)

        #blstm_cell = bLSTMCell(
        #        input_size=d_model,
        #        hidden_size=d_model,
        #        weight_ih_attr=weight_attr, 
        #        weight_hh_attr=weight_attr,
        #        bias_ih_attr=bias_attr,
        #        bias_hh_attr=bias_attr
        #        )
        #self.lstm_layers = nn.RNN(blstm_cell)
        self.lstm_layers = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=nlayers,
                dropout=dropout,
                weight_ih_attr=weight_attr, 
                weight_hh_attr=weight_attr,
                bias_ih_attr=bias_attr,
                bias_hh_attr=bias_attr
                )

        self.output_mapping = Linear(
            d_model, emb_size,
            weight_attr,
            bias_attr)
        self.dropout_1 = paddle.nn.Dropout(dropout)
        #self.dropout_2 = paddle.nn.Dropout(dropout)

        self.tied_nlayers = tied_nlayers

    def forward(self,
                features,
                src_mask=None,
                pe_out=None,
                mems=None):
        """
        src: features, labels - [Batch_size, Segment], Int
        """
        # [Batch_size, Segment, Embedding_Size]
        src = self.fea_embedding(features)
        #print(self.fea_embedding(paddle.to_tensor(0))[0])
        new_mems = []
        output = self.dropout_1(src)
        
        if(mems is None):
            for i in range(self.tied_nlayers):
                output, update_mems = self.lstm_layers(output)
                new_mems.append(update_mems)
        else:
            for i in range(self.tied_nlayers):
                output, update_mems = self.lstm_layers(src, initial_states=mems[i])
                new_mems.append(update_mems)

        #output = self.output_mapping(self.dropout_2(output))
        output = self.output_mapping(output)

        return output, new_mems
