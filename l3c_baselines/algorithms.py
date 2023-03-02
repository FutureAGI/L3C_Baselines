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
from copy import deepcopy
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

    def __init__(self, model, gamma = 0.98):
        self._model = model
        self._target_model = deepcopy(model)
        self._gamma = gamma

    def train(self, states, actions, rewards):
        target_values, _ = self._target_model(states, actions, rewards)
        values, actions_softmax = self._model(states, actions, rewards)
        target_values = paddle.roll(target_values.detach(), -1) + rewards
