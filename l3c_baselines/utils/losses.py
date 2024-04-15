import torch
from torch import nn
from torch.nn import functional as F

def mse_loss_mask(img_out, img_gt, mask = None):
    mse_loss = torch.mean(((img_out - img_gt / 255)) ** 2, dim=[2, 3, 4])
    if mask is not None:
        mse_loss = mse_loss * mask
        sum_mask = torch.sum(mask)
        sum_loss = torch.sum(mse_loss)
        mse_loss = sum_loss / sum_mask
    else:
        mse_loss = torch.mean(mse_loss)

    return mse_loss

def ce_loss_mask(act_out, act_gt, mask = None, gamma=1):
    print("GT:\n", act_gt[0][10:15], "\nOut\n", act_out[0][10:15])
    gt_logits = F.one_hot(act_gt, act_out.shape[-1])
    preds = torch.log(act_out) * ((1.0 - act_out.detach()) ** gamma)
    ce_loss = -torch.mean(preds * gt_logits, dim=-1)
    if mask is not None:
        ce_loss = ce_loss * mask
        sum_mask = torch.sum(mask)
        sum_loss = torch.sum(ce_loss)
        ce_loss = sum_loss / sum_mask
    else:
        ce_loss = torch.mean(ce_loss)

    return ce_loss
