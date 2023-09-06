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
import math
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
"""MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile
Vision Applications" for more details. Code is taken from
https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenet.py
"""


class InvertedResidual(nn.Module):
    """Expand + depthwise + pointwise."""

    def __init__(self, in_planes: int, out_planes: int, expansion: int, stride: int):
        super(InvertedResidual, self).__init__()
        if stride not in [1, 2]:
            raise ValueError(f"stride should be either 1 or 2 instead of {stride}")
        self.stride = stride
        self.is_shortcut = False
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, 
            out_channels=planes, 
            kernel_size=1,
            stride=1, 
            padding=0, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.conv2 = nn.Conv2d(
            in_channels=planes, 
            out_channels=planes, 
            kernel_size=3,
            stride=stride, 
            padding=1, 
            groups=planes, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.conv3 = nn.Conv2d(
            in_channels=planes, 
            out_channels=out_planes, 
            kernel_size=1,
            stride=1, 
            padding=0, 
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(num_features=out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.is_shortcut = True
            conv_module = nn.Conv2d(
                in_channels=in_planes, 
                out_channels=out_planes, 
                kernel_size=1,
                stride=1, 
                padding=0, 
                bias=False
            )
            bn_module = nn.BatchNorm2d(num_features=out_planes)
            if hasattr(bn_module, 'is_imp'):
                bn_module.is_imp = True
            self.shortcut = nn.Sequential(conv_module, bn_module)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out

    def get_flops(self, inp):
        flops = 0

        output = self.conv1(inp)
        active_elements_count = output.shape[0] * reduce(
            lambda x, y: x * y, output.shape[2:])
        flops += (self.conv1.in_channels * self.conv1.out_channels *
                  active_elements_count)
        flops += 4 * output.numel()  # relu + bn
        output = F.relu(output)

        output = self.conv2(output)
        active_elements_count = output.shape[0] * reduce(
            lambda x, y: x * y, output.shape[2:])
        flops += (self.conv2.in_channels *
                  (self.conv2.out_channels // self.conv2.groups) * 9 *
                  active_elements_count)
        flops += 4 * output.numel()  # relu + bn
        output = F.relu(output)

        output = self.conv3(output)
        active_elements_count = output.shape[0] * reduce(
            lambda x, y: x * y, output.shape[2:])
        flops += (self.conv3.in_channels * self.conv3.out_channels *
                  active_elements_count)
        flops += 3 * output.numel()  # bn + residual

        if self.is_shortcut:
            r_output = self.shortcut[0](inp)
            active_elements_count = r_output.shape[0] * reduce(
                lambda x, y: x * y, r_output.shape[2:])
            flops += (self.shortcut[0].in_channels *
                      self.shortcut[0].out_channels * active_elements_count)
            flops += 2 * r_output.numel()

        return flops, output


class MobileNetv2(BaseModel):
    # CFG -> (expansion, out_planes, num_blocks, stride)
    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),  # NOTE: change layers[1] stride 2->1 for CIFAR10
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes, cfg=None):
        super(MobileNetv2, self).__init__()
        if cfg:
            if len(cfg)==0 or len(cfg[0])!=4:
                raise ValueError(f"cfg should be a 4-element list, got {cfg}")
            self.cfg = cfg
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=32, 
            kernel_size=3,
            stride=2,   # NOTE: change conv1 stride 2->1 for CIFAR10
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=32)
        if hasattr(self.bn1, 'is_imp'):
            self.bn1.is_imp = True
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(
            in_channels=self.cfg[-1][1], 
            out_channels=1280, 
            kernel_size=1,
            stride=2, 
            padding=0, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_features=1280)
        if hasattr(self.bn2, 'is_imp'):
            self.bn2.is_imp = True
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes: int):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    # TorchScript doesn't support inheritance, so the superclass method should have
    # name other than 'forward' that can be accessed in a subclass
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def get_flops(self, insize):
        flops = 0
        inp = torch.rand(1, 3, insize, insize).to(self.conv1.weight.device)
        output = self.conv1(inp)
        active_elements_count = output.shape[0] * reduce(
            lambda x, y: x * y, output.shape[2:])
        flops += (self.conv1.in_channels * self.conv1.out_channels * 9 *
                  active_elements_count)
        flops += 4 * output.numel()  # bn + relu
        output = F.relu(output)

        for b in self.layers:
            b_flops, output = b.get_flops(output)
            flops += b_flops

        output = self.conv2(output)
        active_elements_count = output.shape[0] * reduce(
            lambda x, y: x * y, output.shape[2:])
        flops += (self.conv2.in_channels * self.conv2.out_channels *
                  active_elements_count)
        flops += 4 * output.numel()  # bn + relu
        output = F.relu(output)

        ops = output.shape[0] * reduce(lambda x, y: x * y, output.shape[2:])
        output = self.avgpool(output)
        flops += ops * output.numel()

        flops += (1 + self.linear.in_features) * self.linear.out_features
        return flops


def get_mobilenet(model, num_classes, cfg=None):
    """Returns the requested model, ready for training/pruning with the
    specified method.

    :param model: str
    :param method: full or prune
    :param num_classes: int, num classes in the dataset
    :return: A prunable MobileNet model
    """
    if model == 'mobilenetv2':
        net = MobileNetv2(num_classes, cfg)
    return net
