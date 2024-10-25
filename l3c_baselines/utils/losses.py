import torch
from torch import nn
from torch.nn import functional as F

def mse_loss_mask(img_out, img_gt, mask=None, reduce_dim=[0,1]):
    """
    input shape: (B, T, *)
    mask shape: (B, T)
    reduce for dim=1
    """
    dim = img_out.dim()
    assert dim >= 3 and dim == img_gt.dim(), "Input must be at least 3 dimension (bsz, time, *) and the label and prediction must be equal"
    mse_loss = torch.mean(((img_out - img_gt)) ** 2, dim=[i for i in range(2, dim)])
    if mask is not None:
        mse_loss = mse_loss * mask
        if reduce_dim == 1:
            sum_mask = torch.mean(mask)
            sum_loss = torch.mean(mse_loss)
        elif reduce_dim == 0:
            sum_mask = torch.mean(mask, dim=0)
            sum_loss = torch.mean(mse_loss, dim=0)
        mse_loss = sum_loss / sum_mask
    else:
        if reduce_dim == 1:
            mse_loss = torch.mean(mse_loss)
        elif reduce_dim == 0:
            mse_loss = torch.mean(mse_loss, dim=0)

    return mse_loss

def ce_loss_mask(act_out, act_gt, mask = None, gamma=1, reduce_dim=[0,1]):
    """
    input shape: (B, T, H)
    mask shape: (B, T)
    reudce for dim=1
    """
    gt_logits = F.one_hot(act_gt, act_out.shape[-1])
    preds = torch.log(act_out + 1.0e-10) * ((1.0 - act_out) ** gamma)
    ce_loss = -torch.sum(preds * gt_logits, dim=-1)

    if mask is not None:
        ce_loss = ce_loss * mask
        if reduce_dim == 1:
            sum_mask = torch.mean(mask)
            sum_loss = torch.mean(ce_loss)
        elif reduce_dim == 0:
            sum_mask = torch.mean(mask, dim=0)
            sum_loss = torch.mean(ce_loss, dim=0)
        ce_loss = sum_loss / sum_mask
    else:
        if reduce_dim == 1:
            ce_loss = torch.mean(ce_loss)
        elif reduce_dim == 0:
            ce_loss = torch.mean(ce_loss, dim=0)

    return ce_loss

def ent_loss(act_out, reduce_dim=[0,1]):
    """
    input shape: (B, T, H)
    mask shape: (B, T)
    reudce for dim=1
    """
    if reduce_dim == 1:
        return torch.mean(torch.log(act_out + 1.0e-10) * act_out)
    elif reduce_dim == 0:
        return torch.mean(torch.log(act_out + 1.0e-10) * act_out, dim=0)
    else:
        return torch.log(act_out + 1.0e-10) * act_out
