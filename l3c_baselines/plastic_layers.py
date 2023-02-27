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
"""Plasticity Layers"""

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


class FullPlasticLayer(nn.Layer):
    """Full Plastic Layer
        Fully connected layer taking x_in = [batch, L, D_in] as input and give x_out = [batch, L, D_out] as output.
        Input to output projection is linear as x_out = W * x_in + b
        The plasticity in W is given by 
            W = W + m (A x_out * x_in + B x_out + C x_in + D)
        a is calculated by post synaptic signals
            m = sigma (W' * x_out + b')
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 plasticity_rule="mABCD",
                 dopamin_type="neuron",                 #neuron/single
                 dopamine_activation="sigmoid",
                 memory_decay=0.01,
                 memory_decay_thres=5.0,
                 initial_variance=0.10,
                 learning_rate=0.01,
                 is_batch_update=True,
                 rules_attr=None,
                 dopamine_weight_attr=None,
                 dopamine_bias_attr=None):
        super(FullPlasticLayer, self).__init__()
        self.plasticity_rule = plasticity_rule
        self.d_in = input_dim
        self.d_out = output_dim
        self.dopamine_type = dopamine_type
        self.is_batch_update = is_batch_update

        self._dtype = self._helper.get_default_dtype()
        if(learning_rate is None):
            self.learning_rate = paddle.sqrt(paddle.to_tensor(1.0 / output_dim))
        else:
            self.learning_rate = learning_rate
        self.weight_attr = rules_attr
        self.dopamine_weight_attr = dopamine_weight_attr
        self.dopamine_bias_attr = dopamine_bias_attr
        self.dopamine_activation = getattr(
            F, dopamine_activation)

        if(self.plasticity_rule == "mABCD"):
            self.init_parameter_1()
        else:
            raise Exception("Unexpected plasticity rule type:", self.plasticity_rule)

        self.output_bias = self.create_parameter(
                shape=[self.d_out],
                attr=None,
                dtype=self._dtype,
                is_bias=True)

        self.mem_decay_thres = memory_decay_thres
        self.mem_decay = memory_decay
        self.initial_variance = initial_variance

    def plasticity_update(self, i, o, mem):
        # i: [B, L, D_in]
        # o: [B, L, D_out]
        # mem: [B, D_in, D_out]
        # return updated mem:  [B, D_in, D_out]
        if(self.plasticity_rule == "mABCD"):
            if(self.is_batch_update):
                return self.plasticity_update_1_b(i, o, mem)
            else:
                return self.plasticity_update_1_s(i, o, mem)
        else:
            raise Exception("Unexpected plasticity rule type:", self.plasticity_rule)

    def init_parameter_1(self):
        # Parameter initialization for mABCD plasticity rule
        self.w_a = self.create_parameter(
            shape=[self.d_in, self.d_out],
            attr=self.weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.w_b = self.create_parameter(
            shape=[self.d_in, self.d_out],
            attr=self.weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.w_c = self.create_parameter(
            shape=[self.d_in, self.d_out],
            attr=self.weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.w_d = self.create_parameter(
            shape=[self.d_in, self.d_out],
            attr=self.weight_attr,
            dtype=self._dtype,
            is_bias=False)
        if(self.dopamine_type == "neuron"):
            self.dopamine_mapping = Linear(
                self.d_out, self.d_out,
                weight_attr=self.dopamine_weight_attr,
                bias_attr=self.dopamine_bias_attr)
        elif(self.dopamine_type == "single"):
            self.dopamine_mapping = Linear(
                self.d_out, 1,
                weight_attr=self.dopamine_weight_attr,
                bias_attr=self.dopamine_bias_attr)

    def plasticity_update_1_b(self, i, o, mem):
        # i: [B, L, D_in], x_in : [B, L, D_in ,1]
        # o: [B, L, D_out], x_out : [B, L, 1 ,D_out]
        x_in = paddle.unsqueeze(i, 3)
        x_out = paddle.unsqueeze(o, 2)

        # io, ii, oo: [B, L, D_in, D_out]
        io = paddle.matmul(x_in, x_out)
        ii = paddle.repeat_interleave(x_in, self.d_out, axis=3)
        oo = paddle.repeat_interleave(x_out, self.d_in, axis=2)

        # raw_mod: [B, L, D_in, D_out]
        raw_mod = self.w_a * io + self.w_b * ii + self.w_c * oo + self.w_d

        # modulation: [B, L, 1]
        modulation = self.dopamine_activation(self.dopamine_mapping(o) - 1.0)

        # goes back to [B, D_in, D_out]
        mem += self.learning_rate * paddle.mean(raw_mod * paddle.unsqueeze(modulation, 3), axis=1)
        delta_mem = mem - paddle.clip(mem, min=-self.mem_decay_thres, max=self.mem_decay_thres)
        mem -= self.mem_decay * delta_mem

        return mem

    def plasticity_update_1_s(self, i, o, mem):
        # i: [B, D_in], x_in : [B, D_in ,1]
        # o: [B, D_out], x_in : [B, 1 ,D_out]
        x_in = paddle.unsqueeze(i, 2)
        x_out = paddle.unsqueeze(o, 1)

        # io, ii, oo: [B, L, D_in, D_out]
        io = paddle.matmul(x_in, x_out)
        ii = paddle.repeat_interleave(x_in, self.d_out, axis=2)
        oo = paddle.repeat_interleave(x_out, self.d_in, axis=1)

        raw_mod = self.w_a * io + self.w_b * ii + self.w_c * oo + self.w_d
        modulation = self.dopamine_activation(self.dopamine_mapping(o) - 1.0)

        mem += self.learning_rate * paddle.mean(raw_mod * paddle.unsqueeze(modulation, 3), axis=1)
        delta_mem = mem - paddle.clip(mem, min=-self.mem_decay_thres, max=self.mem_decay_thres)
        mem -= self.mem_decay * delta_mem

        return mem

    def forward(self, src, mem):
        """ src: input [B, L, D_in],  tgt: additional output [B, L, D_out]
            mem: memory [B, D_in, D_out]
        """
        out = F.linear(src, mem) + self.output_bias 
        return out

    def update_mem(self, src, tgt, mem):
        """ src: input  [B, L, D_in] 
            tgt: output [B, L, D_out]
            mem: memory [B, D_in, D_out]
        """
        return self.plasticity_update(src, tgt, mem)
