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
                out_mem_key,
                src_mask,
                cache=None):
        mlen = in_mem.shape[1]
        srclen = src.shape[1] + mlen
        alllen = srclen + mlen
        srcs = paddle.concat([in_mem, src, out_mem_key], axis=1)
        if(cache is None):
            tgts = super(RMTLayer, self).forward(srcs, src_mask=src_mask)
        else:
            tgts, incremental_cache = super(RMTLayer, self).forward(srcs, src_mask=src_mask, cache=cache)

        if cache is not None:
            return tgts[:,:mlen], tgts[:,mlen:srclen], tgts[:,srclen:], cache
        else:
            return tgts[:,:mlen], tgts[:,mlen:srclen], tgts[:,srclen:]

### Differences between training & inference!!!!
class RMTEncoder(TransformerEncoder):
    """TransformerEncoder is a stack of N encoder layers."""
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(RMTEncoder, self).__init__(encoder_layer, num_layers, norm)

    def forward(self,
                src,
                mems_in,
                mems_key_out,
                caches=None):

        if caches is not None:
            new_caches = []

        batch_size, mem_len, d_model = mems_in.shape
        assert mems_key_out.shape[1] == mems_in.shape[1], "in and output mem must be equal"
        _, src_len, _ = src.shape
        alllen = src_len + 2 * mem_len

        src_mask = paddle.tensor.triu(
                    paddle.full(shape=[alllen, alllen],
                    fill_value=-np.inf,
                    dtype=src.dtype), 1)

        output = src
        m_in = mems_in
        m_out = mems_key_out
        for i, mod in enumerate(self.layers):
            # NOTE: Support different memory each layer or same memory each layer
            if caches is not None:
                m_in, output, m_out, new_cache = mod(m_in, output, m_out, src_mask, cache=caches[i])
                new_caches.append(new_cache)
            else:
                m_in, output, m_out = mod(m_in, output, m_out, src_mask)

        if self.norm is not None:
            output = self.norm(output)
            mems_out = self.norm(m_out)
        else:
            mems_out = m_out

        if caches is not None:
            return output, mems_out, new_caches
        else:
            return output, mems_out

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
        self.pos_embeddings_in = nn.Embedding(mem_len, nhidden)
        self.pos_embeddings_out = nn.Embedding(mem_len, nhidden)

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
            mems = paddle.zeros((bsz, self.mem_len, self.n_hidden), dtype=paddle.get_default_dtype())

        src_len = self.mem_len * 2 + seg
        type_inputs = paddle.zeros((bsz, seg), dtype=paddle.get_default_dtype())
        token_embs = self.word_embeddings(inputs)
        pos_embs = paddle.unsqueeze(self.pos_embeddings.weight[:seg], axis=0)
        pos_emb_in = paddle.unsqueeze(self.pos_embeddings_in.weight, axis=0)
        pos_emb_out = paddle.unsqueeze(self.pos_embeddings_out.weight, axis=0)

        features = token_embs + pos_embs
        mems_in = mems + pos_emb_in 
        mems_key_out = mems + pos_emb_out

        out, n_mems = self.encoder(
            features,
            mems_in,
            mems_key_out
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
