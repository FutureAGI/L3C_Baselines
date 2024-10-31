import sys
import torch
import torch.distributed as dist
from types import SimpleNamespace
from copy import deepcopy
from dateutil.parser import parse
from collections import defaultdict
from l3c_baselines.utils import log_debug, log_warn


def incremental_update_mean_var(mean, var, count, 
                                batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count

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

    def append_with_safety(self, device, count=None, **kwargs):
        """
        Count being regarded as the number of samples behind each of the value
        if value is an array, then it is regarded as the number of samples behind each of the value
        """
        zeroflag = False
        if(count is None):
            count = torch.Tensor([1]).to(device)
        assert count.numel() == 1, "count must have only one element"

        for key, value in kwargs.items():
            if torch.isinf(value).any() or torch.isnan(value).any():
                print(f"[WARNING] 'Device:{device}' stating '{key}' suffering prediction loss = NAN/INF, fill with 0")
                zeroflag = True
        safe_stats = dict()
        if zeroflag:
            for key, value in kwargs.items():
                safe_stats[key] = torch.zeros_like(value)
        else:
            for key, value in kwargs.items():
                safe_stats[key] = value.detach()

        for key, value in safe_stats.items():
            if(key not in self._data):
                raise KeyError(f"Key {key} not registered in Statistics class")
            
            #loss matrix dim is [2,T//downsample_length], first row is position_wise mean, second row is variance.
            gathered_tensors = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
            gathered_counts = [torch.zeros_like(count) for _ in range(dist.get_world_size())]

            # gather values from all devices
            dist.all_gather(gathered_tensors, value.data)
            dist.all_gather(gathered_counts, count.data)

            #If device num is 8, self._data[key] has 8 elements, each element is a tensor with shape [2,T//downsample_length]
            #Each element can be a length-1, length-2 tensors
            self._data[key].extend(gathered_tensors)
            self._count[key].extend(gathered_counts)

    def _stat(self, key):
        value = torch.stack(self._data[key], dim=0)
        counts = torch.stack(self._count[key], dim=0)

        counts = counts.view([-1] + [1] * (value.ndim - 1))

        sum_cnt = torch.clip(torch.sum(counts), min=1)
        x_mean = torch.sum(value * counts, dim=0, keepdim=False) / sum_cnt
        x2_mean = torch.sum(value ** 2 * counts, dim=0, keepdim=False) / sum_cnt

        var = torch.sqrt(x2_mean - x_mean ** 2)

        return x_mean, var, sum_cnt

    def __call__(self, reset=True):
        stat_res = dict()
        for key in self.keys:
            mean,var,cnt = self._stat(key)
            if(mean.numel() < 2):
                mean = float(mean)
                var = float(var)
            stat_res[key] = {"mean":mean,"var":var, 'cnt': cnt}
        if(reset):
            self.reset()
        return stat_res
