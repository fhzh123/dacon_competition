# Import modules
import random
import numpy as np

# Import PyTorch
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

def spm_encoding(title_list, content_list, spm_, args, total=True):
    if total:
        return [[args.bos_idx] + spm_.EncodeAsIds(title) + [args.sep_idx] + \
                spm_.EncodeAsIds(content) + [args.eos_idx] \
                for title, content in zip(title_list, content_list)]
    else:
        title_indices = [[args.bos_idx] + spm_.EncodeAsIds(title) + [args.eos_idx] \
                         for title in title_list]
        content_indices = [[args.bos_idx] + spm_.EncodeAsIds(content) + [args.eos_idx] \
                           for content in content_list]
        return title_indices, content_indices

class WarmupLinearSchedule(LambdaLR):
    """ 
        Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))