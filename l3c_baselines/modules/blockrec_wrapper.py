#!/usr/bin/env python
# A wrapper that wraps the model with block-recurrence

# coding=utf8
# File: models.py
import sys
import random
import torch
import numpy
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from l3c_baselines.utils import ce_loss_mask, mse_loss_mask, img_pro, img_post
from l3c_baselines.utils import format_cache

class BlockRecurrentWrapper(nn.Module):
    """
    Wrapping a temporal modeler with a memory cache to make it block-recurrent
    """
    def __init__(self, temporal_module, memory_length, memory_type='kv'):
        """
        Memory_Type: "kv", "mem"
        """
        super().__init__()

        self.reset()
        self.temporal_module = temporal_module
        self.mem_len = memory_length
        self.memory_type = memory_type.lower()

    def reset(self):
        # This will clear the memory and the cache
        self.memory = None
        
    def merge_memory_in_cache(self, cache):
        if(self.memory_type == "kv"):
            if(cache is not None and self.memory is not None):
                new_cache = []
                for mem, ca in zip(self.memory, cache):
                    new_cache.append(torch.cat((mem, ca), dim=1))
            elif(self.memory is not None):
                new_cache = self.memory
            elif(cache is not None):
                new_cache = cache
            else:
                new_cache = None
            return new_cache
        elif(self.memory_type == "mem"):
            if(cache is not None):
                new_cache = cache
            else:
                new_cache = self.memory
            return new_cache

    def update_memory_cache(self, cache):
        # Updates the Memory and Cache
        # For KV cache, in case the memory + cache > 2 * memory_length, we update the memory
        # Else, we keep the cache and the memory
        def copy_cache(cache):
            if(isinstance(cache, torch.abstract.Tensor)):
                return cache.clone().detach()
            if(isinstance(cache, list)):
                return [copy_cache(c) for c in cache]
            elif(isinstance(cache, tuple)):
                return tuple([copy_cache(c) for c in cache])
            else:
                raise Exception("Unknown type of cache")

        if(self.memory_type == "kv"):
            if(cache is not None):
                self.memory = [c[:, -self.mem_len:].clone().detach() for c in cache]
            else:
                self.memory = None
            return None
        elif(self.memory_type == "mem"):
            # Just update the memory and the cache
            self.memory = copy_cache(cache)
        else:
            raise Exception(f"No such memory type: {self.memory_type}")

    def update_cache_only(self, cache):
        if(self.memory_type == 'kv'):
            if(self.memory is None):
                return cache
            else:
                new_cache = []
                for m,c in zip(self.memory, cache):
                    m_len = m.shape[1]
                    new_cache.append(c[m_len:])
                return new_cache
        elif(self.memory_type == "mem"):
            return cache
        else:
            raise Exception(f"No such memory type: {self.memory_type}")
            
    def forward(self, src, cache=None, need_cache=False, verbose=True, checkpoints_density=-1, update_memory=True):
        # when update memory = False, inference won't update the memory, but will update the cache
        output, new_cache = self.temporal_module.forward(
                src, 
                cache=self.merge_memory_in_cache(cache), 
                need_cache=True, 
                checkpoints_density=checkpoints_density)
        if(update_memory):
            new_cache = self.update_memory_cache(new_cache)
        elif(need_cache):
            new_cache = self.update_cache_only(new_cache)
        else:
            new_cache = None
        #print(new_cache[0][0].shape, torch.sum(new_cache[0][0]), new_cache[0][1].shape, torch.sum(new_cache[0][1]))

        return output, new_cache
