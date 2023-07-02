# MIT License
#
# Copyright (c) 2023 Transmute AI Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm import tqdm_notebook
from ..algorithms import BaseAlgorithm

__all__ = [
    'BaseQuantization',
    'StraightThrough',
    'RoundSTE',
    'Conv2dFunctor',
    'LinearFunctor',
    'FoldBN',
]


class BaseQuantization(BaseAlgorithm):
    """Base class for quantization algorithms."""

    def __init__(self, **kwargs):
        super(BaseQuantization, self).__init__(**kwargs)
        pass

    def quantize(self, model, dataloaders, method, **kwargs):
        pass

    def round_ste(x: torch.Tensor):
        """Implement Straight-Through Estimator for rounding operation."""
        return (x.round() - x).detach() + x

    def get_calib_samples(self, train_loader, num_samples):
        """Get calibration-set samples for finetuning weights and clipping
        parameters."""
        calib_data = []
        for batch in train_loader:
            calib_data.append(batch[0])
            if len(calib_data) * batch[0].size(0) >= num_samples:
                break
        return torch.cat(calib_data, dim=0)[:num_samples]

    def absorb_bn(self, module, bn_module):
        w = module.weight.data
        if module.bias is None:
            zeros = torch.Tensor(module.out_channels).zero_().type(w.type())
            module.bias = nn.Parameter(zeros)
        b = module.bias.data
        invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
        w.mul_(invstd.view(w.size(0), 1, 1, 1).expand_as(w))
        b.add_(-bn_module.running_mean).mul_(invstd)

        if bn_module.affine:
            w.mul_(bn_module.weight.data.view(w.size(0), 1, 1, 1).expand_as(w))
            b.mul_(bn_module.weight.data).add_(bn_module.bias.data)

        bn_module.register_buffer('running_mean',
                                  torch.zeros(module.out_channels).cuda())
        bn_module.register_buffer('running_var',
                                  torch.ones(module.out_channels).cuda())
        bn_module.register_parameter('weight', None)
        bn_module.register_parameter('bias', None)
        bn_module.affine = False

    def is_bn(self, m):
        return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)

    def is_absorbing(self, m):
        return (isinstance(m, nn.Conv2d) and m.groups == 1) or isinstance(
            m, nn.Linear)

    def search_absorbe_bn(self, model):
        prev = None
        for m in model.children():
            if self.is_bn(m) and self.is_absorbing(prev):
                m.absorbed = True
                self.absorb_bn(prev, m)
            self.search_absorbe_bn(m)
            prev = m


class StraightThrough(nn.Module):
    """Used to place an identity function in place of a non-differentail
    operator for gradient calculation."""

    def __int__(self):
        super().__init__()
        pass

    def forward(self, input):
        return input


class RoundSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Conv2dFunctor:

    def __init__(self, conv2d):
        self.conv2d = conv2d

    def __call__(self, *input, weight, bias):
        res = torch.nn.functional.conv2d(*input, weight, bias,
                                         self.conv2d.stride,
                                         self.conv2d.padding,
                                         self.conv2d.dilation,
                                         self.conv2d.groups)
        return res


class LinearFunctor:

    def __init__(self, linear):
        self.linear = linear

    def __call__(self, *input, weight, bias):
        res = torch.nn.functional.linear(*input, weight, bias)
        return res


# TODO : To migrate all BN-layer folding function calls to the ones defined inside BaseQuantization class
class FoldBN:
    """Used to fold batch norm to prev linear or conv layer which helps reduce
    comutational overhead during quantization."""

    def __init__(self):
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
        bn_module.running_var = bn_module.weight.data**2

    def is_bn(self, m):
        return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)

    def is_absorbing(self, m):
        return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)

    def search_fold_and_remove_bn(self, model: nn.Module):
        """Method to recursively search for batch norm layers, absorb them into
        the previous linear or conv layers, and set it to an identity layer."""
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
