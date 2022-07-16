
import torch
import torch.nn as nn
import torch.nn.init as init

from trailmet.algorithms.quantize.q_utils import StraightThrough
from ..algorithms import BaseAlgorithm

class BaseQuantization(BaseAlgorithm):
    """base class for quantization algorithms"""
    def __init__(self, **kwargs):
        super(BaseQuantization, self).__init__(**kwargs)
        pass

    def quantize(self, model, dataloaders, method, **kwargs):
        pass
    

    def round_ste(x: torch.Tensor):
        """
        Implement Straight-Through Estimator for rounding operation.
        """
        return (x.round() - x).detach() + x

    def lp_loss(pred, tgt, p=2.0, reduction='none'):
        """
        loss function measured in L_p Norm
        """
        if reduction == 'none':
            return (pred-tgt).abs().pow(p).sum(1).mean()
        else:
            return (pred-tgt).abs().pow(p).mean()

    class StraightThrough(nn.Module):
        def __int__(self):
            super().__init__()
            pass

        def forward(self, input):
            return input
    
    class FoldBN():
        """class to help fold batch norm to prev layer activations"""
        def __init__(self):
            super().__init__()
            pass

        def _fold_bn(self, conv_module, bn_module):
            w = conv_module.weight.data
            y_mean = bn_module.running_mean
            y_var = bn_module.running_var
            safe_std = torch.sqrt(y_var + bn_module.eps)
            w_view = (conv_module.out_channels, 1, 1, 1)
            if bn_module.affine:
                weight = w * (bn_module.weight / safe_std).view(w_view)
                beta = bn_module.bias - bn_module.weight * y_mean / safe_std
                if conv_module.bias is not None:
                    bias = bn_module.weight * conv_module.bias / safe_std + beta
                else:
                    bias = beta
            else:
                weight = w / safe_std.view(w_view)
                beta = -y_mean / safe_std
                if conv_module.bias is not None:
                    bias = conv_module.bias / safe_std + beta
                else:
                    bias = beta
            return weight, bias


        def fold_bn_into_conv(self, conv_module, bn_module):
            w, b = self._fold_bn(conv_module, bn_module)
            if conv_module.bias is None:
                conv_module.bias = nn.Parameter(b)
            else:
                conv_module.bias.data = b
            conv_module.weight.data = w
            # set bn running stats
            bn_module.running_mean = bn_module.bias.data
            bn_module.running_var = bn_module.weight.data ** 2


        def reset_bn(self, module: nn.BatchNorm2d):
            if module.track_running_stats:
                module.running_mean.zero_()
                module.running_var.fill_(1-module.eps)
                # we do not reset numer of tracked batches here
                # self.num_batches_tracked.zero_()
            if module.affine:
                init.ones_(module.weight)
                init.zeros_(module.bias)


        def is_bn(self, m):
            return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


        def is_absorbing(self, m):
            return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)


        def search_fold_and_remove_bn(self, model: nn.Module):
            model.eval()
            prev = None
            for n, m in model.named_children():
                if self.is_bn(m) and self.is_absorbing(prev):
                    self.fold_bn_into_conv(prev, m)
                    # set the bn module to straight through
                    setattr(model, n, StraightThrough())
                elif self.is_absorbing(m):
                    prev = m
                else:
                    prev = self.search_fold_and_remove_bn(m)
            return prev


        def search_fold_and_reset_bn(self, model: nn.Module):
            model.eval()
            prev = None
            for n, m in model.named_children():
                if self.is_bn(m) and self.is_absorbing(prev):
                    self.fold_bn_into_conv(prev, m)
                    # reset_bn(m)
                else:
                    self.search_fold_and_reset_bn(m)
                prev = m
