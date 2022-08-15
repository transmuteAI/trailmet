import torch.nn as nn
from torch.hub import load_state_dict_from_url
from .base_model import BaseModel
from collections import defaultdict
import numpy as np
import math

import torch


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class channel_selection(nn.Module):
 
    def __init__(self, num_channels):
    
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
    
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,))
        output = input_tensor[:, selected_index, :, :]
        return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activ = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activ(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activ(out)

        return out

class BottleneckNs(nn.Module):
    expansion = 4

    def __init__(self, inplanes,cfg, planes,stride=1, downsample=None):
        super(BottleneckNs, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.activ = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activ(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activ(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activ(out)

        return out

class ResNetCifar(BaseModel):
    def __init__(self, block, layers, width=1, num_classes=1000, insize=32):
        super(ResNetCifar, self).__init__()
        self.inplanes = 16
        self.insize = insize
        self.layers_size = layers
        self.num_classes = num_classes
        self.width = width
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.prev_module=defaultdict()
        self.prev_module[self.bn1]=None
        self.activ = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16 * width, layers[0])
        self.layer2 = self._make_layer(block, 32 * width, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * width, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(64 * width, num_classes)
        self.init_weights()

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
            conv_module = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
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
        x = self.activ(x)

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
    def __init__(self, block, layers, width=1, num_classes=1000, produce_vectors=False, init_weights=True, insize=32):
        super(ResNet, self).__init__()
        self.layers_size = layers
        self.num_classes = num_classes
        self.insize = insize
        self.produce_vectors = produce_vectors
        self.block_type = block.__class__.__name__
        self.inplanes = 64
        if insize<128:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activ = nn.ReLU(inplace=True)
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
                downs = next(b.downsample.children()) if b.downsample is not None else None

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv_module = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
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
        x = self.activ(x)
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


class ResnetNs(ResNetCifar):
    def __init__(self, block , depth=164, dataset='cifar10', cfg=None , num_classes = 10):
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        n = (depth - 2) // 9
        block = block

        if cfg is None:
            # Construct config variable.
            cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
            cfg = [item for sub_list in cfg for item in sub_list]
        super(ResnetNs, self).__init__(block = Bottleneck , layers = [4,4,4])
        del self.layer1,self.layer2,self.layer3


        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                              bias=False)
        self.layer1 = self._make_layer_cfg(block, 16, n, cfg = cfg[0:3*n])
        self.layer2 = self._make_layer_cfg(block, 32, n, cfg = cfg[3*n:6*n], stride=2)
        self.layer3 = self._make_layer_cfg(block, 64, n, cfg = cfg[6*n:9*n], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.select = channel_selection(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        if dataset == 'cifar10':
            self.fc = nn.Linear(cfg[-1], 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(cfg[-1], 100)
        else:
            self.fc = nn.Linear(cfg[-1],num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer_cfg(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, cfg[0:3],planes,  stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, cfg[3*i: 3*(i+1)],planes))

        return nn.Sequential(*layers)


def make_ns_resnet(num_classes , cfg = None):
    model = ResnetNs(BottleneckNs , num_classes = num_classes , cfg = cfg)
    return model


def make_wide_resnet(num_classes, insize):
    model = ResNetCifar(BasicBlock, [4, 4, 4], width=12, num_classes=num_classes, insize=insize)
    return model

def make_resnet20(num_classes, insize):
    model = ResNetCifar(BasicBlock, [3, 3, 3], width=1, num_classes=num_classes, insize=insize)
    return model

def make_resnet32(num_classes, insize):
    model = ResNetCifar(BasicBlock, [5, 5, 5], width=1, num_classes=num_classes, insize=insize)
    return model

def make_resnet50(num_classes, insize):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, insize=insize)
    return model

def make_resnet56(num_classes, insize):
    model = ResNetCifar(BasicBlock, [9, 9, 9], width=1, num_classes=num_classes, insize=insize)
    return model

def make_resnet18(num_classes, insize):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, insize=insize)
    return model

def make_resnet101(num_classes, insize):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, insize=insize)
    return model

def make_resnet110(num_classes, insize):
    model = ResNetCifar(BasicBlock, [18, 18, 18], width=1, num_classes=num_classes, insize=insize)
    return model

def make_resnet152(num_classes, insize):
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, insize=insize)
    return model

def get_resnet_model(model, num_classes, insize, pretrained):
    """Returns the requested model, ready for training/pruning with the specified method.
    :param model: str, either wrn or r50
    :param num_classes: int, num classes in the dataset
    :return: A prunable ResNet model
    """
    if model == 'wrn':
        net = make_wide_resnet(num_classes, insize)
        pretrained_weights = None

    elif model == 'resnet18':
        net = make_resnet18(num_classes, insize)
        pretrained_weights = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    elif model == 'resnet20':
        net = make_resnet20(num_classes, insize)
        pretrained_weights = None
    elif model == 'resnet32':
        net = make_resnet32(num_classes, insize)
        pretrained_weights =  None
    elif model == 'resnet50':
        net = make_resnet50(num_classes, insize)
        pretrained_weights = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
    elif model == 'resnet56':
        net = make_resnet56(num_classes, insize)
        pretrained_weights = None 
    elif model == 'resnet101':
        net = make_resnet101(num_classes, insize)
        pretrained_weights = "https://download.pytorch.org/models/resnet101-cd907fc2.pth"
    elif model == 'resnet110':
        net = make_resnet110(num_classes, insize)
        pretrained_weights = None
    elif model == 'resnet152':
        net = make_resnet152(num_classes, insize)
        pretrained_weights = "https://download.pytorch.org/models/resnet152-f82ba261.pth"
    if pretrained:
        if pretrained_weights != None:
            weights = load_state_dict_from_url(pretrained_weights, progress=True)
            net.load_state_dict(weights, strict = False)
        else:
            print("pretrained weights not available")
    return net
