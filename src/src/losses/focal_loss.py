import torch
import torch.nn.functional as F

def focal_loss(pred, gt, alpha=2, beta=4):
    pos = gt.eq(1).float()
    neg = gt.lt(1).float()

    pos_loss = -torch.log(pred + 1e-6) * torch.pow(1 - pred, alpha) * pos
    neg_loss = -torch.log(1 - pred + 1e-6) * torch.pow(pred, alpha) * torch.pow(1 - gt, beta) * neg

    num_pos = pos.sum()
    loss = (pos_loss.sum() + neg_loss.sum()) / (num_pos + 1e-6)
    return loss
