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
from .plastic_layers import FullPlasticLayer
from .paddle.nn.layer.transformer import _convert_attention_mask
from .plasmer_block import FullPlasticLayer


def segmentation(segment_size, *args):
    """
    Truncate Squences of Size [B, L, D] to [B, T, D], [B, T, D]
    args: list of arguments in shape of [B, L, D]
    """
    if(len(args) < 0):
        raise Exception("Must have at least 1 sequence to truncate")
    b_idx = 0
    seg_idx = 0
    L = args[0].shape[1]
    while b_idx < L:
        seg_idx += 1
        e_idx = min(b_idx + segment_size, L)
        if(e_idx <= b_idx):
            return
        yield tuple([b_idx, e_idx] + [arg[:, b_idx:e_idx] for arg in args])
        b_idx = e_idx

class SoftmaxSAC(object):
    # Basic Soft AC algorithm for softmax

    def __init__(self, model, configs):
        self._gamma = configs["gamma"]
        self._alpha = configs["alpha"]
        self._seg_size = configs["segment_size"]
        self._tau = configs["tau"]

        self._model = paddle.DataParallel(model)
        self._target_model = paddle.DataParallel(deepcopy(model))

        #lr = paddle.optimizer.lr.NoamDecay(d_model=1.0e-3, warmup_steps=1000)
        #lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=1.0e-3, T_max=1000, eta_min=1.0e-4)
        #lr = paddle.optimizer.lr.CyclicLR(base_learning_rate=2.0e-3, max_learning_rate=5.0e-5, step_size_up=10)
        #lr = paddle.optimizer.lr.InverseTimeDecay(learning_rate=5.0e-3, gamma=0.25, last_epoch=-1)
        lr = paddle.optimizer.lr.ExponentialDecay(learning_rate=1.0e-3, gamma=0.99)
        self.opt = paddle.optimizer.SGD(learning_rate=lr, parameters=model.parameters(),
                  grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
            )
        self.opt = fleet.distributed_optimizer(adam)

    def sync_model_param(self, model, target_model):
        for i, param in enumerate(model.params):
            target_model.params[i] += self.tau * (model.params[i] - target_model.params[i])

    def train(self, states, actions, rewards):
        t_stat = paddle.to_tensor(states)
        t_act = paddle.to_tensor(actions)
        t_rew = paddle.to_tensor(rewards)
        
        # target_values: Batch * L * ActDim
        # Actions_probs: Batch * L * ActDim, Sum last dimension = 1.0
        target_mem = None
        mem = None
        self._model.train()
        sum_loss = 0.0
        for b_idx, e_idx, r_stat, r_act, r_rew in segementation(self.seg_size, t_stat, t_act, t_rew):
            target_values, target_acts, l2, target_mem = self._target_model(r_stat, r_act, r_rew, target_mem)
            values, acts, l2, mem = self._model(r_stat, r_act, r_rew, mem)
            log_acts = paddle.log(acts)
            target_values = paddle.reduce_sum(acts.detach() * (paddle.roll(target_values.detach(), -1) * gamma + 
                rewards - self._alpha * log_acts.detach()), axis=-1)
            loss = ( 1.0 * paddle.reduce_sum(paddle.reduce_sum(acts * log_acts, axis=-1)) + 
                1.0 * paddle.reduce_sum(paddle.square(paddle.target_values[:,:,:-1] - values[:,:,:-1])) +
                0.2 * l2)
            loss.backward()
            #debug_print_grad(model)
            self._opt.step()
            self._opt.clear_grad()
            mem = detached_memory(mem)
            target_mem = detached_memory(target_mem)
            sum_loss += loss.numpy()[0]
        self.sync_model_param(self._model, self._target_model)

        return sum_loss

class MaxEntropy(object):
    # Basic Soft AC algorithm for softmax

    def __init__(self, model, configs):
        self._seg_size = configs["segment_size"]

        self._model = paddle.DataParallel(model)

        lr = paddle.optimizer.lr.ExponentialDecay(learning_rate=1.0e-3, gamma=0.99)
        self.opt = paddle.optimizer.AdamW(learning_rate=lr, parameters=model.parameters(),
            grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0))
        self.opt = fleet.distributed_optimizer(adam)

    def train(self, features, labels):
        feas = paddle.to_tensor(features)
        labs = paddle.to_tensor(labels)
        
        # target_values: Batch * L * ActDim
        # Actions_probs: Batch * L * ActDim, Sum last dimension = 1.0
        target_mem = None
        mem = None
        self._model.train()
        sum_loss = 0.0
        for b_idx, e_idx, r_stat, r_act, r_rew in segementation(self.seg_size, t_stat, t_act, t_rew):
            target_values, target_acts, l2, target_mem = self._target_model(r_stat, r_act, r_rew, target_mem)
            values, acts, l2, mem = self._model(r_stat, r_act, r_rew, mem)
            log_acts = paddle.log(acts)
            target_values = paddle.reduce_sum(acts.detach() * (paddle.roll(target_values.detach(), -1) * gamma + 
                rewards - self._alpha * log_acts.detach()), axis=-1)
            loss = ( 1.0 * paddle.reduce_sum(paddle.reduce_sum(acts * log_acts, axis=-1)) + 
                1.0 * paddle.reduce_sum(paddle.square(paddle.target_values[:,:,:-1] - values[:,:,:-1])) +
                0.2 * l2)
            loss.backward()
            #debug_print_grad(model)
            self._opt.step()
            self._opt.clear_grad()
            mem = detached_memory(mem)
            target_mem = detached_memory(target_mem)
            sum_loss += loss.numpy()[0]
        self.sync_model_param(self._model, self._target_model)

        return sum_loss
