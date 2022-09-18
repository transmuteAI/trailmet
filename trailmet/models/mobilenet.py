"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
"""

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.inp = in_planes
        self.oup = out_planes
        self.exp = expand_ratio

        hidden_dim = round(in_planes * expand_ratio)
        # self.identity = stride == 1 and in_planes == out_planes

        self.conv1 = nn.Conv2d(in_planes, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
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
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetv2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 1],   # NOTE: change stride 2 -> 1 for CIFAR
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.width_mult = width_mult
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        # NOTE: change conv1 stride 2 -> 1 for CIFAR
        self.conv1 = nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        block = InvertedResidual
        self.layers = self._make_layers(in_planes=32, block=block)
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280       
        self.conv2 = nn.Conv2d(320, output_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(output_channel, num_classes)

    def _make_layers(self, in_planes, block=InvertedResidual):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfgs:
            out_channel = _make_divisible(out_planes*self.width_mult, 4 if self.width_mult == 0.1 else 8)
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(block(in_planes, out_channel, stride, expansion))
                in_planes = out_planes
        return nn.Sequential(*layers)  

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out  


def get_mobilenet(model, num_classes):
    """Returns the requested model, ready for training/pruning with the specified method.
    :param model: str
    :param num_classes: int, num classes in the dataset
    :return: A MobileNet model
    """
    assert model in __all__, '{model} model not present'
    if model == 'mobilenetv2':
        net = MobileNetv2(num_classes)
    return net