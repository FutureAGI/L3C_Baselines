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

import sys
import paddle
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_parallel import ColumnParallelLinear, RowParallelLinear
from paddle.fluid import layers
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.layer.transformer import TransformerEncoderLayer, TransformerEncoder, _convert_attention_mask

class RMT2MemEncLayer(TransformerEncoderLayer):
    def forward(self, src, src_mask=None, cache=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        # Add cache for encoder for the usage like UniLM
        if cache is None:
            src = self.self_attn(src, src, src, src_mask)
        else:
            src, incremental_cache = self.self_attn(
                src, src, src, src_mask, cache
            )

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        return src if cache is None else (src, incremental_cache)

class RMT2Layer(TransformerEncoderLayer):
    """
    Implementation for Recurrent Memory Transformer (RMT)
    Bulatov, Aydar, Yury Kuratov, and Mikhail Burtsev. "Recurrent memory transformer." Advances in Neural Information Processing Systems 35 (2022): 11079-11091.
    """
    def __init__(self, *args, **kwargs):
        super(RMT2Layer, self).__init__(*args, **kwargs)

    def forward(self,
                in_mem,
                src,
                cache=None):
        mlen = in_mem.shape[1]
        slen = src.shape[1]
        srcs = paddle.concat([in_mem, src], axis=1)
        src_mask = paddle.tensor.triu(
                    paddle.full(shape=[slen + mlen, slen + mlen],
                    fill_value=-np.inf,
                    dtype=src.dtype), diagonal=1)
        src_mask[:mlen, :] = 0
        if(cache is None):
            tgts = super(RMT2Layer, self).forward(srcs, src_mask=src_mask)
        else:
            tgts, incremental_cache = super(RMT2Layer, self).forward(srcs, src_mask=src_mask, cache=cache)

        if cache is not None:
            return tgts[:, :mlen], tgts[:, mlen:(slen + mlen)], cache
        else:
            return tgts[:, :mlen], tgts[:, mlen:(slen + mlen)]

### Differences between training & inference!!!!
class RMT2Encoder(TransformerEncoder):
    """TransformerEncoder is a stack of N encoder layers."""
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(RMT2Encoder, self).__init__(encoder_layer, num_layers, norm)

    def forward(self,
                src,
                mems_in,
                caches=None):

        if caches is not None:
            new_caches = []

        batch_size, mem_len, d_model = mems_in[0].shape
        _, src_len, _ = src.shape
        alllen = src_len + mem_len

        output = src
        mems_out = []
        for i, mod in enumerate(self.layers):
            # NOTE: Support different memory each layer or same memory each layer
            if caches is not None:
                m_out, output, new_cache = mod(mems_in[i], output, cache=caches[i])
                new_caches.append(new_cache)
            else:
                m_out, output = mod(mems_in[i], output)
            mems_out.append(m_out)

        if self.norm is not None:
            output = self.norm(output)

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

class AutoRegressiveRMT2(nn.Layer):
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
        super(AutoRegressiveRMT2, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, nhidden)
        self.pos_embeddings = nn.Embedding(seg_len, nhidden)
        self.pos_embeddings_in = [nn.Embedding(mem_len, nhidden) for _ in range(nlayer)]

        self.encoder = RMT2Encoder(RMT2Layer(nhidden, nhead, 4 * nhidden), nlayer)
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
            mems = [paddle.zeros((bsz, self.mem_len, self.n_hidden), dtype=paddle.get_default_dtype()) for _ in range(self.n_layer)]

        type_inputs = paddle.zeros((bsz, seg), dtype=paddle.get_default_dtype())
        token_embs = self.word_embeddings(inputs)
        pos_embs = paddle.unsqueeze(self.pos_embeddings.weight[:seg], axis=0)
        pos_emb_in = [paddle.unsqueeze(self.pos_embeddings_in[i].weight, axis=0) for i in range(self.n_layer)]

        features = token_embs + pos_embs
        mems_pos = [mems[i] + pos_emb_in[i] for i in range(self.n_layer)]

        out, n_mems = self.encoder(
            features,
            mems_pos,
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
