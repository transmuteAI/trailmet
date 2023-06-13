# +
from functools import reduce

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from .base_model import BaseModel
# -

from collections import defaultdict

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

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
    
    def get_flops(self, inp):
        flops = 0
        output = self.conv1(inp)
        active_elements_count = output.shape[0] * reduce(lambda x, y: x * y, output.shape[2:])
        flops += self.conv1.in_channels*self.conv1.out_channels*9*active_elements_count
        flops += 4 * output.numel() #relu + bn
        output = self.conv2(output)
        active_elements_count = output.shape[0] * reduce(lambda x, y: x * y, output.shape[2:])
        flops += self.conv2.in_channels*self.conv2.out_channels*9*active_elements_count
        flops += 5 * output.numel() #relu + bn + residual

        if self.downsample is not None:
            r_output = self.downsample[0](inp)
            active_elements_count = r_output.shape[0] * reduce(lambda x, y: x * y, r_output.shape[2:])
            flops += self.downsample[0].in_channels*self.downsample[0].out_channels*active_elements_count
            flops += 2 * r_output.numel()
        return flops

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
        self.prev_module=defaultdict()
        self.prev_module[self.bn1]=None
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

        #assert block is Bottleneck
        if block is Bottleneck:
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
    
    def get_flops(self, insize):
        flops = 0
        inp = torch.rand(1, 3, insize, insize).to(self.conv1.weight.device)
        output = self.conv1(inp)
        active_elements_count = output.shape[0] * reduce(lambda x, y: x * y, output.shape[2:])
        flops += self.conv1.in_channels*self.conv1.out_channels*9*active_elements_count
        output = self.bn1(output)
        flops += 2*output.numel()
        output = self.activ(output)
        flops += 2*output.numel()
        output = self.maxpool(output)
        flops += 8*output.numel()
        
        for l_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for b in l_block:
                flops += b.get_flops(output)
                output = b(output)
        
        ops = output.shape[-1]*output.shape[-2]
        output = self.avgpool(output)
        flops += ops*output.numel()
        
        flops += (1 + self.fc.in_features)*self.fc.out_features
        return flops


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

def make_resnet26(num_classes, insize):
    model = ResNet(BasicBlock, [3, 3, 3, 3], num_classes=num_classes, insize=insize)
    return model


def make_resnet34(num_classes, insize):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, insize=insize)
    return model


def make_resnet18(num_classes, insize):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, insize=insize)
    return model

model = make_resnet18(1000, 64)
model


def make_resnet10(num_classes, insize):
    model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, insize=insize)
    return model


def make_resnet56(num_classes, insize):
    model = ResNetCifar(BasicBlock, [9, 9, 9], width=1, num_classes=num_classes, insize=insize)
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
    elif model == 'resnet34':
        net = make_resnet34(num_classes, insize)
        pretrained_weights = None
    elif model == 'resnet26':
        net = make_resnet26(num_classes, insize)
        pretrained_weights = None
    elif model == 'resnet10':
        net = make_resnet10(num_classes, insize)
        pretrained_weights = None
    if pretrained:
        if pretrained_weights != None:
            weights = load_state_dict_from_url(pretrained_weights, progress=True)
            net.load_state_dict(weights, strict = False)
        else:
            print("pretrained weights not available")
    return net
