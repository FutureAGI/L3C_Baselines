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
from paddle.fluid import layers
from paddle.nn import Linear
import paddle.incubate as incubate
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.layer.rnn import RNNBase, RNN, BiRNN

class AutoRegressiveLSTM(nn.Layer):
    """LSTM Classifier"""
    def __init__(self, 
            vocab_size, 
            nhidden, 
            nlayer, 
            dropout=0.10, 
            weight_attr=None, 
            bias_attr=None):
        super(AutoRegressiveLSTM, self).__init__()

        self.fea_embedding = paddle.nn.Embedding(vocab_size, nhidden)

        self.lstm_layers = nn.LSTM(
                input_size=nhidden,
                hidden_size=nhidden,
                num_layers=nlayer,
                dropout=dropout,
                weight_ih_attr=weight_attr, 
                weight_hh_attr=weight_attr,
                bias_ih_attr=bias_attr,
                bias_hh_attr=bias_attr
                )

        self.output_mapping = Linear(
            nhidden, vocab_size,
            weight_attr,
            bias_attr)
        self.dropout_1 = paddle.nn.Dropout(dropout)

    def forward(self,
                features,
                mems=None):
        """
        src: features, labels - [Batch_size, Segment], Int
        """
        # [Batch_size, Segment, Embedding_Size]
        src = self.fea_embedding(features)

        output = self.dropout_1(src)
        
        if(mems is None):
            output, update_mems = self.lstm_layers(output)
        else:
            output, update_mems = self.lstm_layers(src, initial_states=mems)

        #output = self.output_mapping(self.dropout_2(output))
        output = self.output_mapping(output)

        return output, update_mems

if __name__=="__main__":
    vocab = 200
    model = AutoRegressiveLSTM(vocab, 128, 2)
    input_token = paddle.randint(low=0, high=vocab, shape=[64, 256], dtype="int32")
    outputs, mems = model.forward(input_token)
    print("1st output", outputs.shape)
    outputs, mems = model.forward(input_token, mems=mems)
    print("2nd output", outputs.shape)
