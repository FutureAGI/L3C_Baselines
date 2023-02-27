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
from plastic_layers import FullPlasticLayer
from paddle.nn.layer.transformer import _convert_attention_mask
from plasmer_block import FullPlasticLayer

class PRNNCell(nn.RNNCellBase):
    def __init__(self,
                 input_size,
                 hidden_size,
                 activation="tanh",
                 weight_ih_attr=None,
                 weight_hh_attr=None,
                 bias_ih_attr=None,
                 bias_hh_attr=None,
                 name=None):
        super(PRNNCell, self).__init__()
        if hidden_size <= 0:
            raise ValueError(
                "hidden_size of {} must be greater than 0, but now equals to {}"
                .format(self.__class__.__name__, hidden_size))
        std = 1.0 / math.sqrt(hidden_size)
        self.bias_ih = self.create_parameter(
            (hidden_size, ),
            bias_ih_attr,
            is_bias=True,
            default_initializer=I.Uniform(-std, std))
        self.bias_hh = self.create_parameter(
            (hidden_size, ),
            bias_hh_attr,
            is_bias=True,
            default_initializer=I.Uniform(-std, std))

        self.plastic_i2h = FullPlasticLayer(
                input_size,
                hidden_size,
                plasticity_rule="mABCD",
                dopamine_activation="sigmoid",
                memory_decay=0.01,
                memory_decay_thres=5.0,
                initial_variance=0.10,
                learning_rate=0.01,
                rules_attr=self.weight_attr,
                dopamine_weight_attr=weight_hh_attr,
                dopamine_bias_attr=bias_hh_attr)

        self.plastic_h2h = FullPlasticLayer(
                hidden_size,
                hidden_size,
                plasticity_rule="mABCD",
                dopamine_activation="sigmoid",
                memory_decay=0.01,
                memory_decay_thres=5.0,
                initial_variance=0.10,
                learning_rate=0.01,
                rules_attr=self.weight_attr,
                dopamine_weight_attr=weight_hh_attr,
                dopamine_bias_attr=bias_hh_attr)

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.activation = paddle.tanh

    def forward(self, inputs, states=None):
        if states is None:
            states = self.get_initial_states(inputs, self.state_shape)
        pre_h, mem_h = states
        i2h = paddle.matmul(inputs, self.weight_ih, transpose_y=True)
        if self.bias_ih is not None:
            i2h += self.bias_ih
        h2h = self.plastic_fc(inputs, mem_h)
        if self.bias_hh is not None:
            h2h += self.bias_hh
        h = self._activation_fn(i2h + h2h)
        updated_mem = self.plastic_fc.update_mem(inputs, h, mem_h)
        return h, (h, updated_mem)

    @property
    def state_shape(self):
        return (self.hidden_size, (self.hidden_size, self.hidden_size))

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.activation != "tanh":
            s += ', activation={activation}'
        return s.format(**self.__dict__)

class PRNNlassifier(nn.Layer):
    """plastic Classifier"""

    def __init__(self, d_model, emb_size, nlayers=1, scale_factor=5.0, dropout=0.10, tied_nlayers=1, weight_attr=None, bias_attr=None):
        super(PRNNClassifier, self).__init__()

        self.fea_embedding = paddle.nn.Embedding(emb_size, d_model)

        self.prnn_layers=nn.LayerList()
        for i in range(nlayers):
            prnn_cell = prnnCell(
                    input_size=d_model,
                    hidden_size=d_model,
                    weight_ih_attr=weight_attr, 
                    weight_hh_attr=weight_attr,
                    bias_ih_attr=bias_attr,
                    bias_hh_attr=bias_attr
                    )
            self.prnn_layers.append(nn.RNN(blstm_cell))

        self.output_mapping = Linear(
            d_model, emb_size,
            weight_attr,
            bias_attr)
        self.dropout_1 = paddle.nn.Dropout(dropout)
        self.dropout_2 = paddle.nn.Dropout(dropout)

        self.nlayers = nlayers

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
            for i in range(self.nlayers):
                output, update_mems = self.prnn_layers(output)
                new_mems.append(update_mems)
        else:
            for i in range(self.nlayers):
                output, update_mems = self.prnn_layers(src, initial_states=mems[i])
                new_mems.append(update_mems)

        output = self.output_mapping(self.dropout_2(output))

        return output, new_mems
