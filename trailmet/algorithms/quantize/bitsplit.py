


import torch
import torch.nn as nn
from trailmet.algorithms.quantize.quantize import BaseQuantization


class BitSplit(BaseQuantization):
    def __init__(self, model: nn.Module, dataloaders, **kwargs):
        super(BitSplit, self).__init__(**kwargs)
        self.model = model
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        