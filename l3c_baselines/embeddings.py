#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""Unified Embeddings."""

import numpy as np
import paddle
from paddle.distributed.fleet.meta_parallel import VocabParallelEmbedding
import paddle.nn as nn

from transformer_block import MultiHeadAttention


class UnifiedEmbeddings(nn.Layer):

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_position_embeddings=256,
                 type_vocab_size=2,
                 role_type_size=None,
                 turn_type_size=None,
                 dropout=0.1,
                 use_type=True,
                 use_role=False,
                 use_turn=False,
                 norm=None,
                 pe_style="abs",
                 clamp_len=0,
                 weight_attr=None):
        super(UnifiedEmbeddings, self).__init__()

        # token embeddings
        self.token_embedding = VocabParallelEmbedding(vocab_size, hidden_size, weight_attr=weight_attr)

        # pos embeddings
        if pe_style == "rel":
            inv_freq = 1 / (10000 ** (paddle.arange(0.0, hidden_size, 2.0) / hidden_size))
            self.register_buffer("inv_freq", inv_freq)
        else:
            self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size, weight_attr=weight_attr)

        # type embeddings
        self.use_type = use_type
        if self.use_type:
            self.type_embedding = nn.Embedding(type_vocab_size, hidden_size, weight_attr=weight_attr)

        # role embeddings
        self.use_role = use_role
        if self.use_role:
            self.role_embedding = nn.Embedding(role_type_size, hidden_size, weight_attr=weight_attr)

        # turn embeddings
        self.use_turn = use_turn
        if self.use_turn:
            self.turn_embedding = nn.Embedding(turn_type_size, hidden_size, weight_attr=weight_attr)

        self.norm = norm
        self.pe_style = pe_style
        self.clamp_len = clamp_len

        self.dropout = nn.Dropout(dropout, mode="upscale_in_train")

    def forward(self,
                token_ids,
                pos_ids,
                type_ids,
                role_ids,
                turn_ids,
                input_mask,
                aux_emb):
        """Generate input embeddings of Transformer

        Args:
            tokens_ids: represents the token id of each token, shape is [batch_size, max_seq_len, 1]
            pos_ids: represents the position of each token, shape is [batch_size, max_seq_len, 1]
            type_ids: represents the type of each token, shape is [batch_size, max_seq_len, 1]
            input_mask: represents the attention masking mastrix in each Transformer blocks,
                shape is [batch_size, max_seq_len, max_seq_len]
            aux_emb: represents the auxiliary input embeddings of Transformer.

        Returns:
            A Tuple contains the input embeddings and the attention masking matrix of Transformer.
        """
        token_emb_out = self.token_embedding(token_ids)

        if self.pe_style == "abs":
            pos_emb_out = self.pos_embedding(pos_ids)
            emb_out = token_emb_out + pos_emb_out
        elif self.pe_style == "rel":
            emb_out = token_emb_out
        else:
            raise NotImplementedError(f"Unsupported position encodings: {self.pe_style}.")

        if self.use_type:
            type_emb_out = self.type_embedding(type_ids)
            emb_out = emb_out + type_emb_out
        if self.use_role:
            role_emb_out = self.role_embedding(role_ids)
            emb_out = emb_out + role_emb_out
        if self.use_turn:
            turn_emb_out = self.turn_embedding(turn_ids)
            emb_out = emb_out + turn_emb_out

        # concat auxiliary memory embeddings
        if aux_emb is not None:
            emb_out = paddle.concat([aux_emb, emb_out], axis=1)

        if self.norm is not None:
            emb_out = self.norm(emb_out)

        emb_out = self.dropout(emb_out)

        if input_mask is not None:
            # generate n-head self-attention mask
            attn_bias = paddle.scale(x=input_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
            attn_bias = paddle.unsqueeze(attn_bias, [1])
        else:
            attn_bias = None

        return emb_out, attn_bias

    def get_pe_out(self, src, mems=None, caches=None):
        if self.pe_style == "rel":
            if isinstance(mems, (list, tuple)):
                mem = mems[0]
            else:
                mem = mems
            klen = qlen = paddle.shape(src)[1]
            if isinstance(mem, paddle.Tensor):
                klen += paddle.shape(mem)[1]
            elif isinstance(mem, MultiHeadAttention.StaticCache):
                klen += paddle.shape(mem.k)[2]
            if caches is not None:
                klen += paddle.shape(caches[0].k)[2]
            # NOTE:
            # only support bidirectional relative position encoding now
            # range: [-qlen + 1, klen - 1]
            # truncated range: [-rlen, rlen - 1]
            #
            # TODO: # unidirectional relative position encoding
            # range: [0, klen - 1]
            # truncated range: [0, rlen - 1]
            pos_seq = paddle.arange(klen - 1, -qlen, -1, dtype=self.inv_freq.dtype)
            if self.clamp_len > 0:
                pos_seq = paddle.clip(pos_seq, min=-self.clamp_len, max=self.clamp_len)
            sinusoid_inp = paddle.outer(pos_seq, self.inv_freq) # [K, D]
            pe_out = paddle.concat([paddle.sin(sinusoid_inp), paddle.cos(sinusoid_inp)], axis=1)
            return pe_out
        else:
            return None
