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
import numpy
from .utils import debug_print_norm

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
                 dopamine_activation="tanh",
                 activation="gelu",
                 memory_decay=0.02,
                 memory_decay_thres=5.0,
                 initial_variance=0.10,
                 learning_rate=None,
                 rules_attr=None,
                 dopamine_weight_attr=None,
                 dopamine_bias_attr=None):
        super(FullPlasticLayer, self).__init__()
        self.plasticity_rule = plasticity_rule
        self.d_in = input_dim
        self.d_out = output_dim

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
        self.activation = getattr(
            F, activation)

        if(self.plasticity_rule == "mABCD"):
            self.init_parameter_1()
        elif(self.plasticity_rule == "A"):
            self.init_parameter_2()
        elif(self.plasticity_rule is None):
            pass
        else:
            raise Exception("Unexpected plasticity rule type:", self.plasticity_rule)

        self.mem_decay_thres = memory_decay_thres
        self.mem_decay = memory_decay
        self.initial_variance = initial_variance

    def plasticity_update(self, i, o, mem):
        # i: [B, L, D_in]
        # o: [B, L, D_out]
        # mem: [B, D_in, D_out]
        # return updated mem:  [B, D_in, D_out]
        if(self.plasticity_rule == "mABCD"):
            return self.plasticity_update_1(i, o, mem)
        elif(self.plasticity_rule == "A"):
            return self.plasticity_update_2(i, o, mem)
        elif(self.plasticity_rule is None):
            return mem
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
        self.dopamine_mapping = Linear(
            self.d_out, 1,
            weight_attr=self.dopamine_weight_attr,
            bias_attr=self.dopamine_bias_attr)

    def init_parameter_2(self):
        # Parameter initialization for mABCD plasticity rule
        self.w_a = self.create_parameter(
            shape=[self.d_in, self.d_out],
            attr=self.weight_attr,
            dtype=self._dtype,
            is_bias=False)

    def plasticity_update_1(self, i, o, mem):
        # i: [B, L, D_in], x_in : [B, L, D_in ,1]
        # o: [B, L, D_out], x_out : [B, L, 1 ,D_out]
        #debug_print_norm("before", mem)
        x_in = paddle.unsqueeze(i.detach(), 3)
        x_out = paddle.unsqueeze(o.detach(), 2)

        # modulation: [B, L, 1, 1]
        modulation = paddle.unsqueeze(self.dopamine_activation(self.dopamine_mapping(o)), axis=-1)
        x_out = modulation * x_out

        # io, ii, oo: [B, D_in, D_out]
        io = paddle.mean(x_in * x_out, axis=1)
        ii = paddle.mean(modulation * x_in, axis=1)
        oo = paddle.mean(x_out, axis=1)
        xx = paddle.mean(modulation, axis=1)

        # raw_mod: [B, D_in, D_out]
        raw_mod = self.w_a * io + self.w_b * ii + self.w_c * oo + self.w_d * xx

        # goes back to [B, D_in, D_out]
        update_mem = mem + self.learning_rate * raw_mod
        update_mem = self.regularize_mem(update_mem)

        return update_mem

    def plasticity_update_2(self, i, o, mem):
        x_in = paddle.unsqueeze(i.detach(), 3)
        x_out = paddle.unsqueeze(o.detach(), 2)

        io = paddle.mean(paddle.matmul(x_in, x_out), axis=1)
        update_mem = mem + self.learning_rate * self.w_a * io
        update_mem = self.regularize_mem(update_mem)

        return update_mem

    def regularize_mem(self, mem):
        return  (1.0 - self.mem_decay) * mem + self.mem_decay * paddle.clip(mem, -self.mem_decay_thres, self.mem_decay_thres)

    def forward(self, src, mem):
        """ src: input [B, L, D_in],  tgt: additional output [B, L, D_out]
            mem: memory [B, D_in, D_out]
        """
        out = self.activation(F.linear(src, mem)) 
        return out

    def update_mem(self, src, tgt, mem):
        """ src: input  [B, L, D_in] 
            tgt: output [B, L, D_out]
            mem: memory [B, D_in, D_out]
        """
        return self.plasticity_update(src.detach(), tgt.detach(), mem)

class PlasticLayer2(nn.Layer):
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
                 rw_activation=("sigmoid", "softsign"),
                 f_activation=("sigmoid", "sigmoid"),
                 activation="gelu",
                 initial_variance=0.10,
                 learning_rate=None,
                 weight_attr=None,
                 bias_attr=None):
        super(PlasticLayer2, self).__init__()
        self.d_in = input_dim
        self.d_out = output_dim

        self._dtype = self._helper.get_default_dtype()
        if(learning_rate is None):
            self.learning_rate = paddle.sqrt(paddle.to_tensor(1.0 / output_dim))
        else:
            self.learning_rate = learning_rate
        self.weight_attr = weight_attr
        self.bias_attr = bias_attr
        self.rw_activation_i = getattr(
            F, rw_activation[0])
        self.rw_activation_o = getattr(
            F, rw_activation[1])
        self.f_activation_i = getattr(
            F, f_activation[0])
        self.f_activation_o = getattr(
            F, f_activation[1])
        self.activation = getattr(
            F, activation)

        self.init_parameter()
        self.initial_variance = initial_variance

    def plasticity_update(self, i, o, mem):
        # i: [B, L, D_in]
        # o: [B, L, D_out]
        # mem: [B, D_in, D_out]
        # return updated mem:  [B, D_in, D_out]

        # [B, L, D_in / D_out]
        x_in = self.rw_activation_i(paddle.matmul(i, self.w_i) + self.b_i)
        x_out = self.rw_activation_o(paddle.matmul(o, self.w_o) + self.b_o)

        f_in = self.f_activation_i(paddle.matmul(i, self.w_f_i) + self.b_f_i)
        f_out = self.f_activation_o(paddle.matmul(o, self.w_f_o) + self.b_f_o)

        # [B, D_in, D_out]
        delta_w = paddle.mean(paddle.unsqueeze(x_in, 3) * paddle.unsqueeze(x_out, 2), axis=1)
        delta_m = paddle.mean(paddle.unsqueeze(f_in, 3) * paddle.unsqueeze(f_out, 2), axis=1)
        debug_print_norm(delta_w, "delta_w")
        debug_print_norm(delta_m, "delta_m")
        
        return (1.0 - delta_m) * mem + self.learning_rate * delta_w

    def init_parameter(self):
        # Parameter initialization for mABCD plasticity rule
        self.w_i = self.create_parameter(
            shape=[self.d_in, self.d_in],
            attr=self.weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.b_i = self.create_parameter(
            shape=[self.d_in,],
            attr=self.bias_attr,
            dtype=self._dtype,
            is_bias=True)
        self.w_o = self.create_parameter(
            shape=[self.d_out, self.d_out],
            attr=self.weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.b_o = self.create_parameter(
            shape=[self.d_out,],
            attr=self.bias_attr,
            dtype=self._dtype,
            is_bias=True)

        self.w_f_i = self.create_parameter(
            shape=[self.d_in, self.d_in],
            attr=self.weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.b_f_i = self.create_parameter(
            shape=[self.d_in,],
            attr=self.bias_attr,
            dtype=self._dtype,
            is_bias=True)
        self.w_f_o = self.create_parameter(
            shape=[self.d_out, self.d_out],
            attr=self.weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.b_f_o = self.create_parameter(
            shape=[self.d_out,],
            attr=self.bias_attr,
            dtype=self._dtype,
            is_bias=True)

    def forward(self, src, mem):
        """ src: input [B, L, D_in],  tgt: additional output [B, L, D_out]
            mem: memory [B, D_in, D_out]
        """
        out = self.activation(F.linear(src, mem)) 
        return out

    def update_mem(self, src, tgt, mem):
        """ src: input  [B, L, D_in] 
            tgt: output [B, L, D_out]
            mem: memory [B, D_in, D_out]
        """
        return self.plasticity_update(src.detach(), tgt.detach(), mem)
