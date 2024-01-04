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


class RMTLayer(TransformerEncoderLayer):
    """
    Implementation for Recurrent Memory Transformer (RMT)
    Bulatov, Aydar, Yury Kuratov, and Mikhail Burtsev. "Recurrent memory transformer." Advances in Neural Information Processing Systems 35 (2022): 11079-11091.
    """
    def __init__(self, *args, **kwargs):
        super(RMTLayer, self).__init__(*args, **kwargs)

    def forward(self,
                in_mem,
                src,
                cache=None):

        mlen = in_mem.shape[1]
        srclen = src.shape[1] + mlen
        alllen = srclen + mlen
        srcs = paddle.concat([in_mem, src, in_mem], axis=1)
        src_mask = paddle.tensor.triu(
                    paddle.full(shape=[alllen, alllen],
                    fill_value=-np.inf,
                    dtype=src.dtype), 1)
        if(cache is None):
            tgts = super(RMTLayer, self).forward(srcs, src_mask=src_mask)
        else:
            tgts, incremental_cache = super(RMTLayer, self).forward(srcs, src_mask=src_mask, cache=cache)

        if cache is not None:
            return tgts[:,mlen:srclen], tgts[:,srclen:], cache
        else:
            return tgts[:,mlen:srclen], tgts[:,srclen:]

### Differences between training & inference!!!!
class RMTEncoder(TransformerEncoder):
    """TransformerEncoder is a stack of N encoder layers."""
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(RMTEncoder, self).__init__(encoder_layer, num_layers, norm)

    def forward(self,
                src,
                mems,
                caches=None):

        output = src
        if caches is not None:
            new_caches = []

        batch_size, mem_len, d_model = mems[0].shape
        _, src_len, _ = src.shape

        attn_mask = paddle.ones([batch_size, src_len, src_len], dtype=src.dtype)
        attn_mask = paddle.triu(attn_mask)
        attn_mask = paddle.concat(
            [paddle.zeros([batch_size, src_len, mem_len], dtype=src.dtype),
                attn_mask], axis=-1)
        attn_mask = paddle.concat(
            [paddle.zeros([batch_size, mem_len, mem_len + src_len], dtype=src.dtype),
                attn_mask], axis=1)
        attn_bias = paddle.unsqueeze(paddle.scale(attn_mask, scale=-1.0e4, bias=0.0), axis=1)

        all_outputs = [output]
        assert len(mems) == len(self.layers), "Memory do not match layers, Exit"

        all_outputs = []
        n_mems = []
        for i, mod in enumerate(self.layers):
            # NOTE: Support different memory each layer or same memory each layer
            if caches is not None:
                output, n_mem, new_cache = mod(mems[i], output, cache=caches[i])
                new_caches.append(new_cache)
            else:
                output, n_mem = mod(mems[i], output)
            all_outputs.append(output)
            n_mems.append(n_mem)

        if self.norm is not None:
            output = self.norm(output)

        if caches is not None:
            return output, n_mems, new_caches
        else:
            return output, n_mems

    def gen_cache(self, x, do_zip=False):
        """Generates cache for faster decoding step by step.

        The generated cache is a list, and each element in it is Cache or StaticCache
        produced by `TransformerLayer.gen_cache`. See `TransformerLayer.gen_cache`
        for more details. If `do_zip` is True, apply `zip` on these tuples to get
        a list with two elements.
       """
        cache = [layer.gen_cache(x) for layer in self.layers]
        if do_zip:
            cache = list(zip(*cache))
        return cache

class AutoRegressiveRMT(nn.Layer):
    """RMT Encoder Model"""

    def __init__(self,
            vocab_size,
            nhidden,
            nlayer, 
            nhead, 
            seg_len=256,
            mem_len=256,
            weight_attr=None, 
            bias_attr=None):
        super(AutoRegressiveRMT, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, nhidden)
        self.pos_embeddings = nn.Embedding(seg_len, nhidden)

        self.encoder = RMTEncoder(RMTLayer(nhidden, nhead, 4 * nhidden), nlayer)
        self.mem_len = mem_len
        self.n_layer = nlayer
        self.n_hidden = nhidden

        self.output_mapping = nn.Linear(
            nhidden, vocab_size,
            weight_attr,
            bias_attr)

    def forward(self, inputs, mems=None):
        """
        src: features, labels - [Batch_size, Segment], Int
        """
        # [Batch_size, Segment, Embedding_Size]
        bsz, seg = inputs.shape

        if(mems is None):
            mems = [paddle.zeros((bsz, self.mem_len, self.n_hidden), dtype=paddle.get_default_dtype()) for i in range(self.n_layer)]

        src_len = self.mem_len * 2 + seg
        type_inputs = paddle.zeros((bsz, seg), dtype=paddle.get_default_dtype())
        token_embs = self.word_embeddings(inputs)
        pos_embs = paddle.unsqueeze(self.pos_embeddings.weight[:seg], axis=0)

        features = token_embs + pos_embs

        out, n_mems = self.encoder(
            features,
            mems=mems,
        )

        outputs = self.output_mapping(out)

        return outputs, n_mems

if __name__=="__main__":
    vocab = 200
    model = AutoRegressiveRMT(vocab, 128, 2, 8, seg_len=256, mem_len=128)
    input_token = paddle.randint(low=0, high=vocab, shape=[64, 256], dtype="int32")
    outputs, mems = model.forward(input_token)
    print("1st output", outputs.shape)
    outputs, mems = model.forward(input_token, mems=mems)
    print("2nd output", outputs.shape)
