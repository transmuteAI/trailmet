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
import torch
import torch.nn as nn

# from .layers import ModuleInjection, PrunableBatchNorm2d
from .base_model import BaseModel
import numpy as np
"""MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks: Mobile Networks for
Classification, Detection and Segmentation" for more details. Code is taken
from
https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenetv2.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardBinaryConv(nn.Module):

    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand(
            (self.number_of_weights, 1)) * 0.001,
                                    requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(
            torch.mean(
                torch.mean(torch.abs(real_weights), dim=3, keepdim=True),
                dim=2,
                keepdim=True,
            ),
            dim=1,
            keepdim=True,
        )
        # print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = (binary_weights_no_grad.detach() -
                          cliped_weights.detach() + cliped_weights)
        #         print(binary_weights.shape)
        y = F.conv2d(x,
                     binary_weights,
                     stride=self.stride,
                     padding=self.padding)

        return y


class BinaryActivation(nn.Module):

    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        # out_e1 = (x^2 + 2*x)
        # out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(
            torch.float32) + (x * x + 2 * x) * (1 - mask1.type(torch.float32))
        out2 = out1 * mask2.type(
            torch.float32) + (-x * x + 2 * x) * (1 - mask2.type(torch.float32))
        out3 = out2 * mask3.type(
            torch.float32) + 1 * (1 - mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class Block(nn.Module):
    """Expand + depthwise + pointwise."""

    def __init__(self, in_planes, out_planes, expansion, stride, binary=True):
        super(Block, self).__init__()
        self.stride = stride
        self.binary = binary
        planes = expansion * in_planes

        if not self.binary:
            self.act = F.relu
            self.conv1 = nn.Conv2d(in_planes,
                                   planes,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=planes,
                bias=False,
            )
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes,
                                   out_planes,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=False)
            self.bn3 = nn.BatchNorm2d(out_planes)

            self.shortcut = nn.Sequential()
            if stride == 1 and in_planes != out_planes:
                conv_module = nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
                bn_module = nn.BatchNorm2d(out_planes)
                self.shortcut = nn.Sequential(conv_module, bn_module)
        else:
            self.act = BinaryActivation()
            self.conv1 = HardBinaryConv(in_planes,
                                        planes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = HardBinaryConv(planes,
                                        planes,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=1)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = HardBinaryConv(planes,
                                        out_planes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
            self.bn3 = nn.BatchNorm2d(out_planes)

            self.shortcut = nn.Sequential()
            if stride == 1 and in_planes != out_planes:
                conv_module = nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
                bn_module = nn.BatchNorm2d(out_planes)
                self.shortcut = nn.Sequential(conv_module, bn_module)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetv2(BaseModel):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 1),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes=10, num_fp=0):
        super(MobileNetv2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3,
                               32,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.num_fp = num_fp
        self.curr_fp = 0
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320,
                               1280,
                               kernel_size=1,
                               stride=2,
                               padding=0,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            if self.curr_fp < self.num_fp:
                for stride in strides:
                    layers.append(
                        Block(in_planes,
                              out_planes,
                              expansion,
                              stride,
                              binary=False))
                    in_planes = out_planes
                    self.curr_fp += 1
            else:
                for stride in strides:
                    layers.append(
                        Block(in_planes,
                              out_planes,
                              expansion,
                              stride,
                              binary=True))
                    in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        #         out = F.avg_pool2d(out, 7)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def get_mobilenet(num_classes, num_fp=0):
    """Returns the requested model, ready for training/pruning with the
    specified method.

    :param model: str
    :param method: full or prune
    :param num_classes: int, num classes in the dataset
    :return: A prunable MobileNet model
    """
    net = MobileNetv2(num_classes, num_fp)
    return net
