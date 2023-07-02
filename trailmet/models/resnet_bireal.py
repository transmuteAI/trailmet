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
import torch.nn.functional as F
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from .base_model import BaseModel

from collections import defaultdict


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.active = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.active(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.active(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.active = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.active(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.active(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.active(out)

        return out


class ResNetCifar(BaseModel):

    def __init__(self, block, layers, width=1, num_classes=1000, insize=32):
        super(ResNetCifar, self).__init__()
        self.inplanes = 16
        self.insize = insize
        self.layers_size = layers
        self.num_classes = num_classes
        self.width = width
        self.conv1 = nn.Conv2d(3,
                               16,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.prev_module = defaultdict()
        self.prev_module[self.bn1] = None
        self.active = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16 * width, layers[0])
        self.layer2 = self._make_layer(block, 32 * width, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * width, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(64 * width, num_classes)
        self.init_weights()

        assert block is BasicBlock
        prev = self.bn1
        for l_block in [self.layer1, self.layer2, self.layer3]:
            for b in l_block:
                self.prev_module[b.bn1] = prev
                self.prev_module[b.bn2] = b.bn1
                if b.downsample is not None:
                    self.prev_module[b.downsample[1]] = prev
                prev = b.bn2

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv_module = nn.Conv2d(
                self.inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            bn_module = nn.BatchNorm2d(planes * block.expansion)
            if hasattr(bn_module, 'is_imp'):
                bn_module.is_imp = True
            downsample = nn.Sequential(conv_module, bn_module)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.active(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_bn_layers(self):
        bn_layers = []
        for l_blocks in [self.layer1, self.layer2, self.layer3]:
            for b in l_blocks:
                m1, m2 = b.bn1, b.bn2
                bn_layers.append([m1, m2])
        return bn_layers


class ResNet(BaseModel):

    def __init__(
        self,
        block,
        layers,
        width=1,
        num_classes=1000,
        produce_vectors=False,
        init_weights=True,
        insize=32,
    ):
        super(ResNet, self).__init__()
        self.layers_size = layers
        self.num_classes = num_classes
        self.insize = insize
        self.produce_vectors = produce_vectors
        self.block_type = block.__class__.__name__
        self.inplanes = 64
        if insize < 128:
            self.conv1 = nn.Conv2d(3,
                                   64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False)
        else:
            self.conv1 = nn.Conv2d(3,
                                   64,
                                   kernel_size=7,
                                   stride=2,
                                   padding=3,
                                   bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prev_module = defaultdict()
        self.prev_module[self.bn1] = None
        self.active = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 * width, layers[0])
        self.layer2 = self._make_layer(block, 128 * width, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256 * width, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512 * width, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)  # Global Avg Pool
        self.fc = nn.Linear(512 * block.expansion * width, num_classes)
        self.init_weights()

        for l in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for b in l.children():
                downs = (next(b.downsample.children())
                         if b.downsample is not None else None)

        prev = self.bn1
        for l_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for b in l_block:
                self.prev_module[b.bn1] = prev
                self.prev_module[b.bn2] = b.bn1
                self.prev_module[b.bn3] = b.bn2
                if b.downsample is not None:
                    self.prev_module[b.downsample[1]] = prev
                prev = b.bn3

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv_module = nn.Conv2d(
                self.inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            bn_module = nn.BatchNorm2d(planes * block.expansion)
            if hasattr(bn_module, 'is_imp'):
                bn_module.is_imp = True
            downsample = nn.Sequential(conv_module, bn_module)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.active(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feature_vectors = x.view(x.size(0), -1)
        x = self.fc(feature_vectors)

        if self.produce_vectors:
            return x, feature_vectors
        else:
            return x

    def get_bn_layers(self):
        bn_layers = []
        for l_blocks in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for b in l_blocks:
                if self.block_type == 'Bottleneck':
                    m1, m2, m3 = b.bn1, b.bn2, b.bn3
                    bn_layers.append([m1, m2, m3])
                else:
                    m1, m2 = b.bn1, b.bn2
                    bn_layers.append([m1, m2])
        return bn_layers


def make_wide_resnet(num_classes, insize):
    model = ResNetCifar(BasicBlock, [4, 4, 4],
                        width=12,
                        num_classes=num_classes,
                        insize=insize)
    return model


def make_resnet20(num_classes, insize):
    model = ResNetCifar(BasicBlock, [3, 3, 3],
                        width=1,
                        num_classes=num_classes,
                        insize=insize)
    return model


def make_resnet32(num_classes, insize):
    model = ResNetCifar(BasicBlock, [5, 5, 5],
                        width=1,
                        num_classes=num_classes,
                        insize=insize)
    return model


def make_resnet50(num_classes, insize):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   num_classes=num_classes,
                   insize=insize)
    return model


def make_resnet56(num_classes, insize):
    model = ResNetCifar(BasicBlock, [9, 9, 9],
                        width=1,
                        num_classes=num_classes,
                        insize=insize)
    return model


def make_resnet18(num_classes, insize):
    model = ResNet(BasicBlock, [2, 2, 2, 2],
                   num_classes=num_classes,
                   insize=insize)
    return model


def make_resnet101(num_classes, insize):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   num_classes=num_classes,
                   insize=insize)
    return model


def make_resnet110(num_classes, insize):
    model = ResNetCifar(BasicBlock, [18, 18, 18],
                        width=1,
                        num_classes=num_classes,
                        insize=insize)
    return model


def make_resnet152(num_classes, insize):
    model = ResNet(Bottleneck, [3, 8, 36, 3],
                   num_classes=num_classes,
                   insize=insize)
    return model


def get_resnet_model(model, num_classes, insize, pretrained):
    """Returns the requested model, ready for training/pruning with the
    specified method.

    :param model: str, either wrn or r50
    :param num_classes: int, num classes in the dataset
    :return: A prunable ResNet model
    """
    if model == 'wrn':
        net = make_wide_resnet(num_classes, insize)
        pretrained_weights = None

    elif model == 'resnet18':
        net = make_resnet18(num_classes, insize)
        pretrained_weights = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
    elif model == 'resnet20':
        net = make_resnet20(num_classes, insize)
        pretrained_weights = None
    elif model == 'resnet32':
        net = make_resnet32(num_classes, insize)
        pretrained_weights = None
    elif model == 'resnet50':
        net = make_resnet50(num_classes, insize)
        pretrained_weights = 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
    elif model == 'resnet56':
        net = make_resnet56(num_classes, insize)
        pretrained_weights = None
    elif model == 'resnet101':
        net = make_resnet101(num_classes, insize)
        pretrained_weights = (
            'https://download.pytorch.org/models/resnet101-cd907fc2.pth')
    elif model == 'resnet110':
        net = make_resnet110(num_classes, insize)
        pretrained_weights = None
    elif model == 'resnet152':
        net = make_resnet152(num_classes, insize)
        pretrained_weights = (
            'https://download.pytorch.org/models/resnet152-f82ba261.pth')
    if pretrained:
        if pretrained_weights != None:
            weights = load_state_dict_from_url(pretrained_weights,
                                               progress=True)
            net.load_state_dict(weights, strict=False)
        else:
            print('pretrained weights not available')
    return net


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


class BinaryBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 binarise=True):
        super(BinaryBasicBlock, self).__init__()
        self.binarise = binarise
        if self.binarise:
            self.activation = BinaryActivation()
            self.conv = HardBinaryConv(inplanes, planes, stride=stride)
        else:
            self.activation = nn.ReLU(inplace=True)
            self.conv = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = None
        self.bn3 = None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.activation(x)
        out = self.conv(out)
        out = self.bn1(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        return out


class BirealNet(ResNet):

    def __init__(
        self,
        block,
        layers,
        num_fp=0,
        width=1,
        num_classes=1000,
        produce_vectors=False,
        init_weights=True,
        insize=32,
    ):
        super(BirealNet, self).__init__(
            block,
            layers,
            width=1,
            num_classes=num_classes,
            produce_vectors=False,
            init_weights=True,
            insize=32,
        )
        del self.prev_module
        self.num_fp = num_fp
        self.count_fp = 0
        self.inplanes = 64
        self.layer1 = self._make_layer1(block, 64 * width, layers[0])
        self.layer2 = self._make_layer1(block,
                                        128 * width,
                                        layers[1],
                                        stride=2)
        self.layer3 = self._make_layer1(block,
                                        256 * width,
                                        layers[2],
                                        stride=2)
        self.layer4 = self._make_layer1(block,
                                        512 * width,
                                        layers[3],
                                        stride=2)

    def _make_layer1(self, block, planes, blocks, stride=1):
        downsample = None
        if self.count_fp < self.num_fp:
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=stride),
                    conv1x1(self.inplanes, planes * block.expansion),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(
                block(self.inplanes,
                      planes,
                      stride,
                      downsample,
                      binarise=False))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, binarise=False))
            self.count_fp += 1
            return nn.Sequential(*layers)
        else:
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=stride),
                    conv1x1(self.inplanes, planes * block.expansion),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(
                block(self.inplanes, planes, stride, downsample,
                      binarise=True))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, binarise=True))
            return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def make_birealnet18(num_classes, insize, num_fp=0):
    model = BirealNet(
        BinaryBasicBlock,
        [4, 4, 4, 4],
        num_classes=num_classes,
        insize=insize,
        num_fp=num_fp,
    )
    return model


def make_birealnet34(num_classes, insize, num_fp=0):
    model = BirealNet(
        BinaryBasicBlock,
        [6, 8, 12, 6],
        num_classes=num_classes,
        insize=insize,
        num_fp=num_fp,
    )
    return model


def make_birealnet50(num_classes, insize, num_fp=0):
    model = BirealNet(
        BinaryBasicBlock,
        [3, 4, 6, 3],
        num_classes=num_classes,
        insize=insize,
        num_fp=num_fp,
    )
    return model
