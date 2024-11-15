import numpy
import torch

def img_pro(observations):
    return observations / 255

def img_post(observations):
    return observations * 255


def rewards2go(rewards, gamma=0.98):
    """
    returns a future moving average of rewards
    """
    rolled_rewards = rewards.clone()
    r2go = rewards.clone()
    n = max(min(50, -1/numpy.log10(gamma)), 0)
    for _ in range(n):
        rolled_rewards = gamma * torch.roll(rolled_rewards, shifts=-1, dims=1)
        r2go += rolled_rewards
    return r2go

def downsample(x, downsample_length, axis=-1):
    """
    Downsample and get the mean of each segment along a given axis
    """
    if(isinstance(x, torch.Tensor)):
        shape = x.shape
    else:
        shape = numpy.shape(x)
    full_len = shape[axis]
    if(downsample_length >= full_len):
        if(isinstance(x, torch.Tensor)):
            return torch.mean(x, dim=axis, keepdim=True)
        else:
            return numpy.mean(x, axis=axis, keepdims=True)
    trunc_seg = full_len // downsample_length
    trunc_len = trunc_seg * downsample_length

    if(axis == -1):
        new_shape = shape[:axis] + (trunc_seg, downsample_length)
    else:
        new_shape = shape[:axis] + (trunc_seg, downsample_length) + shape[axis + 1:]

    if(isinstance(x, torch.Tensor)):
        ds_x = torch.mean(torch.narrow(x, axis, 0, trunc_len).view(new_shape), dim=axis + 1, keepdim=False)
        if(trunc_len < full_len):
            add_x = torch.mean(torch.narrow(x, axis, trunc_len, full_len - trunc_len), dim=axis, keepdim=True)
            ds_x = torch.cat((ds_x, add_x), dim=axis)
    else:
        slc = [slice(None)] * len(shape)
        slc[axis] = slice(0, trunc_len)
        x = numpy.array(x)
        ds_x = numpy.mean(numpy.reshape(x[tuple(slc)], new_shape), axis=axis + 1, keepdims=False)
        if(trunc_len < full_len):
            slc[axis] = slice(trunc_len, full_len)
            add_x = numpy.mean(x[tuple(slc)], axis=axis, keepdims=True)
            ds_x = numpy.concat((ds_x, add_x), axis=axis)
    return ds_x
