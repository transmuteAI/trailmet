"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
"""

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['mobilenetv2']


class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.inp = in_planes
        self.oup = out_planes
        self.exp = expand_ratio
        
        hidden_planes = round(in_planes * expand_ratio)
        self.identity = stride == 1 and in_planes == out_planes

        self.conv1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)
        self.conv2 = nn.Conv2d(hidden_planes, hidden_planes, kernel_size=3, stride=stride, padding=1, groups=hidden_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_planes)
        self.conv3 = nn.Conv2d(hidden_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride==1 and in_planes!=out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetv2(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetv2, self).__init__()
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 1],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32, block=InvertedResidual)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes, block=InvertedResidual):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfgs:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(block(in_planes, out_planes, stride, expansion))
                in_planes = out_planes
        return nn.Sequential(*layers)  

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out 

def get_mobilenet_model(model_name, num_classes):
    """Returns the requested model, ready for training/compression with the specified method.
    :param model_name: str
    :param num_classes: int, num classes in the dataset
    :return: A MobileNet model
    """
    assert model_name in __all__, '{model} model not present'
    if model_name == 'mobilenetv2':
        net = MobileNetv2(num_classes)
    return net