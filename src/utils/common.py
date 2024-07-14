import random

import torch
import numpy as np

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad and 'normalizer' not in name:
            del state_dict[name]

    return state_dict
