import torch
from torch import nn
from torch.nn import functional as F

def mse_loss_mask(img_out, img_gt, mask=None, reduce="mean"):
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
        if reduce == "mean":
            sum_mask = torch.mean(mask)
            sum_loss = torch.mean(mse_loss)
        else:
            sum_mask = torch.mean(mask, dim=0)
            sum_loss = torch.mean(mse_loss, dim=0)
        mse_loss = sum_loss / sum_mask
    else:
        if reduce == "mean":
            mse_loss = torch.mean(mse_loss)
        else:
            mse_loss = torch.mean(mse_loss, dim=0)

    return mse_loss

def ce_loss_mask(act_out, act_gt, mask = None, gamma=1, reduce="mean"):
    """
    input shape: (B, T, H)
    mask shape: (B, T)
    reudce for dim=1
    """
    gt_logits = F.one_hot(act_gt, act_out.shape[-1])
    preds = torch.log(act_out) * ((1.0 - act_out) ** gamma)
    ce_loss = -torch.sum(preds * gt_logits, dim=-1)

    if mask is not None:
        ce_loss = ce_loss * mask
        if reduce == "mean":
            sum_mask = torch.mean(mask)
            sum_loss = torch.mean(ce_loss)
        else:
            sum_mask = torch.mean(mask, dim=0)
            sum_loss = torch.mean(ce_loss, dim=0)
        ce_loss = sum_loss / sum_mask
    else:
        if reduce == "mean":
            ce_loss = torch.mean(ce_loss)
        else:
            ce_loss = torch.mean(ce_loss, dim=0)

    return ce_loss
