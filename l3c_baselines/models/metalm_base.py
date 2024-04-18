#!/usr/bin/env python
# coding=utf8
# File: models.py
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint  
from modules import ARTransformerStandard
from utils import ce_loss_mask, img_pro, img_post

class LMBase(nn.Module):
    def __init__(self, 
                vocab_size=256,
                hidden_size=768,
                nhead=16,
                max_time_step=1024,
                n_trn_block=12):
        super().__init__()

        self.transformer = ARTransformerStandard(vocab_size, n_trn_block, hidden_size, nhead, max_time_step)


    def forward(self, inputs, cache=None, need_cache=True):
        """
        Inputs: [B, NT]
        Outputs: [B, NT, H]
        """
        
        # Temporal Encoders
        output, new_cache = self.transformer(inputs, cache=cache, need_cache=need_cache)

        return output, new_cache

    def perplexity(self, inputs, outputs, reduce="mean"):
        logits, new_cache = self.forward(inputs, need_cache=False)
        return ce_loss_mask(logits, outputs, gamma=0, reduce=reduce)

    def inference_seg(self, inputs, L, cache=None):
        with torch.no_grad():
            sampled_outputs = inputs
            outputs = inputs
            for _ in range(L):
                logits, cache = self.forward(sampled_outputs, cache=cache, need_cache=True)
                sampled_outputs = torch.multinomial(logits[:, -1], num_samples=1)
                print(sampled_outputs.shape)
                outputs = torch.cat([outputs, sampled_outputs], dim=-1)
        return outputs

if __name__=="__main__":
    model = LMBase()
    inputs = torch.randint(256, (8, 33)) 

    losses = model.perplexity(inputs[:-1], inputs[1:])
    outputs = model.inference_seg(inputs, 4)
    print(losses.shape)
    print(outputs.shape)
