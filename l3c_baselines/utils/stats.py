import sys
import torch
import torch.distributed as dist
from types import SimpleNamespace
from copy import deepcopy
from dateutil.parser import parse
from collections import defaultdict
from .tools import log_debug, log_warn

class DistStatistics(object):
    """
    Provide distributed statistics over GPUs
    """
    def __init__(self, *keys):
        self.keys = keys
        self.reset()

    def reset(self):
        self._data = dict()
        self._count = dict()
        for key in self.keys:
            self._data[key] = []
            self._count[key] = []

    def gather(self, device, count=None, **kwargs):
        """
        Count being regarded as the number of samples behind each of the value
        if value is an array, then it is regarded as the number of samples behind each of the value
        """
        if(count is None):
            fcount = torch.Tensor([1]).to(device)
        elif(isinstance(count, list) or isinstance(count, tuple)):
            fcount = torch.stack(count, dim=0).to(device)
        elif(isinstance(count, torch.Tensor)):
            fcount = count.clone().to(device)
        else:
            fcount = torch.Tensor([count]).to(device)
        
        for key, value in kwargs.items():
            if(key not in self._data):
                log_warn(f"Key {key} not registered in DistStatistics object")
            
            if isinstance(value, list) or isinstance(value, tuple):
                fvalue = torch.stack(value, dim=0).to(device)
            elif(isinstance(value, torch.Tensor)):
                fvalue = value.clone().to(device)
            else:
                fvalue = torch.Tensor(value).to(device)

            assert fcount.numel() == 1 or fcount.numel() == fvalue.numel(), \
                f"dimension mismatch between statistic count {fcount.shape} and value {fvalue.shape}"

            if torch.isinf(fvalue).any() or torch.isnan(fvalue).any():
                log_warn(f"'Device:{device}' stating '{key}' has inf/NaN")
                fvalue = torch.where(torch.isfinite(fvalue), 
                                     fvalue, torch.zeros_like(fvalue))
            
            # Make sure both has the same dimension
            fvalue = fvalue.squeeze()
            if(fcount.ndim > fvalue.ndim):
                fcount = fcount.squeeze()
            while(fcount.ndim < fvalue.ndim):
                fcount = fcount.unsqueeze(-1)
            
            #loss matrix dim is [2,T//downsample_length], first row is position_wise mean, second row is variance.
            gathered_tensors = [torch.zeros_like(fvalue) for _ in range(dist.get_world_size())]
            gathered_counts = [torch.zeros_like(fcount) for _ in range(dist.get_world_size())]

            # gather values from all devices
            dist.all_gather(gathered_tensors, fvalue.data)
            dist.all_gather(gathered_counts, fcount.data)

            #If device num is 8, self._data[key] has 8 elements, each element is a tensor with shape [2,T//downsample_length]
            #Each element can be a length-1, length-2 tensors
            self._data[key].extend(gathered_tensors)
            self._count[key].extend(gathered_counts)

    def _stat(self, key):
        value = torch.stack(self._data[key], dim=0)
        counts = torch.stack(self._count[key], dim=0)

        sum_cnt = torch.clip(torch.sum(counts), min=1)
        x_mean = torch.sum(value * counts, dim=0, keepdim=False) / sum_cnt
        x2_mean = torch.sum(value ** 2 * counts, dim=0, keepdim=False) / sum_cnt

        var = torch.sqrt(x2_mean - x_mean ** 2)

        return x_mean, var, sum_cnt

    def __call__(self, reset=True):
        stat_res = dict()
        for key in self.keys:
            mean,var,cnt = self._stat(key)
            assert cnt.numel() == 1
            cnt = int(cnt)
            if(mean.numel() < 2):
                mean = mean.squeeze().item()
                var = var.squeeze().item()
            else:
                mean = mean.squeeze().tolist()
                var = var.squeeze().tolist()
            stat_res[key] = {"mean":mean,"var":var,'cnt': cnt}
        if(reset):
            self.reset()
        return stat_res
