import torch
from torch import nn
from torch.nn import functional as F

def ent_loss(act_out):
    """
    this returns negative entropy p * log(p) instead of - p * log(p)
    which is directly ready to be minimized, do not add negative sign
    """
    return torch.sum(torch.log(act_out + 1.0e-10) * act_out, dim=-1)

def focal_loss(out, gt, gamma=0):
    gt_logits = F.one_hot(gt, out.shape[-1])
    preds = torch.log(out + 1.0e-10) * ((1.0 - out) ** gamma)
    return -torch.sum(preds * gt_logits, dim=-1)

def metrics(out, gt=None, loss_type='ent', **kwargs):
    if(loss_type == 'mse'):
        assert gt is not None, "Ground Truth Must Be Provided When Using MSE Loss"
        loss_array = torch.mean((out - gt) ** 2, dim=[i for i in range(2, out.ndim)])
    elif(loss_type == 'ce'):
        assert gt is not None, "Ground Truth Must Be Provided When Using Cross Entropy Loss"
        if('gamma' not in kwargs):
            loss_array = focal_loss(out, gt)
        else:
            loss_array = focal_loss(out, gt, kwargs['gamma'])
    elif(loss_type == 'ent'):
        return ent_loss(out)
    else:
        raise ValueError('Unknown loss type {}'.format(loss_type))
    return loss_array
    

def weighted_loss(out, loss_wht=None, reduce_dim=1, need_cnt=False, **kwargs):
    """
    input and ground truth shape: (B, T, *)
    loss_wht shape: (B, T) (or broadcastable), loss weight for each sample

    loss_type: 'mse', 'ce'
    reduce_dim: None - Not Reduced At All
                0 - Only reduce the batch dimension
                1 - Only reduce both the batch and the temporal dimension
    return:
        mse_loss: weighted mean of losses
        sample_cnt: sum of all loss_whts, the same dimension as mse_loss
    """
    
    loss_array = metrics(out, **kwargs)

    if(loss_wht is None):
        loss_wht = torch.full(loss_array.shape[:2], 1.0)
        sample_cnt = loss_array.numel()
    else:
        assert loss_wht.shape[1] == loss_array.shape[1]
        assert loss_wht.shape[0] == loss_array.shape[0] or loss_wht.shape[0] == 1,\
                "loss_wht must be (bsz, time) or (1, time)"
        if(loss_wht.shape[0] == 1):
            sample_cnt = torch.sum(loss_wht) * loss_array.shape[0]
        else:
            sample_cnt = torch.sum(loss_wht)
        loss_array = loss_array * loss_wht

    if(reduce_dim is not None):
        if(reduce_dim==0):
            # verage over batch dimension only
            rdim = [0]
            lambda_ = 1.0 / loss_array.shape[0]
        elif(reduce_dim==1):
            # average over the batch and time dimension
            # must consider the loss weight
            rdim = [0, 1]
            lambda_ = 1.0 / sample_cnt
        else:
            raise ValueError("reduce_dim should be either None, 0 or 1.")
        loss_array = torch.sum(loss_array, dim=rdim) * lambda_

    if(not need_cnt):
        return loss_array
    else:
        return loss_array, sample_cnt
