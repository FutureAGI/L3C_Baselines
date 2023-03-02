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

class SoftmaxSAC(object):
    # Basic Models class
    # Models must inherit from here

    def __init__(self, model):
        self._model = model

    def infer_preprocessing(self, *args):
        raise NotImplementedError()

    def train_preprocessing(self, *args):
        raise NotImplementedError()

    def forward(self, *args):
        preprocessed = self.infer_preprocessing(*args)
        return self._model.forward(*preprocessed)

    def train(self, *args):
        prep_forward, prep_backward = self.train_preprocessing(*args)
        out = self._model.forward(*prep_forward)
        loss = self.loss_func(prep_backward, out)
