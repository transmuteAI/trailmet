
import torch
import torch.nn as nn
import numpy as np
from trailmet.models.resnet import BasicBlock, Bottleneck

class Quantizer(nn.Module):
    def __init__(self, islinear=False, bit_width=8):
        super(Quantizer, self).__init__()
        # self.scale = None
        self.in_scale = None
        self.out_scale = None
        self.signed = islinear
        self.bit_width = bit_width
        self.set_bitwidth(self.bit_width)

    def set_bitwidth(self, bit_width):
        self.bit_width = bit_width
        if self.signed:
            self.max_val = (1 << (self.bit_width - 1)) - 1
            self.min_val = - self.max_val
        else:
            self.max_val = (1 << self.bit_width) - 1
            self.min_val = 0

    def set_scale(self, scale):
        self.set_inscale(scale)
        self.set_outscale(scale)

    def set_inscale(self, in_scale):
        self.in_scale = in_scale
        if isinstance(self.in_scale, (float, np.float32, np.float64)):
            pass
        else:
            self.in_scale = torch.tensor(self.in_scale).view(1, -1, 1, 1)

    def set_outscale(self, out_scale):
        self.out_scale = out_scale
        if isinstance(self.out_scale, (float, np.float32, np.float64)):
            pass
        else:
            self.out_scale = torch.tensor(self.out_scale).view(1, -1, 1, 1)

    def init_quantization(self, x):
        print("x.max", np.max(x))
        print("x.min", np.min(x))
        print("max_val", self.max_val)
        assert(np.min(x)>=0)
    
        circle_detection_queue = [0,]*5
    
        alpha = np.max(np.fabs(x)) / self.max_val
        alpha_old = alpha * 0
        n_iter = 0
        circle_detection_queue[n_iter] = alpha
        while(np.sum(alpha!=alpha_old)):
            q = x / alpha
            q = np.clip(np.round(q), self.min_val, self.max_val)
    
            alpha_old = alpha
            alpha = np.sum(x*q) / np.sum(q*q)
    
            if alpha in circle_detection_queue:
                break
            n_iter += 1
            circle_detection_queue[n_iter%5] = alpha
        return alpha

    def forward(self, x):
        if self.in_scale is None:
            assert(self.out_scale is None)
            return x
        if not isinstance(self.in_scale, (float, np.float32, np.float64)):
            self.in_scale = self.in_scale.to(x.device)
        if not isinstance(self.out_scale, (float, np.float32, np.float64)):
            self.out_scale = self.out_scale.to(x.device)
        return torch.clamp(torch.round(x/self.in_scale), self.min_val, self.max_val) * self.out_scale

class QBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, basic_block: BasicBlock):
        super().__init__()
        self.quant1 = Quantizer()
        self.conv1 = basic_block.conv1
        self.bn1 = basic_block.bn1
        self.activ = basic_block.activ
        self.quant2 = Quantizer()
        self.conv2 = basic_block.conv2
        self.bn2 = basic_block.bn2
        self.downsample = basic_block.downsample
        self.stride = basic_block.stride
        
    def forward(self, x):
        residual = x
        x = self.quant1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activ(out)
        out = self.quant2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activ(out)

        return out


class QBottleneck(nn.Module):
    expansion = 4
    def __init__(self, bottleneck: Bottleneck):
        super().__init__()
        self.quant1 = Quantizer()
        self.conv1 = bottleneck.conv1
        self.bn1 = bottleneck.bn1
        self.quant2 = Quantizer()
        self.conv2 = bottleneck.conv2
        self.bn2 = bottleneck.bn2
        self.quant3 = Quantizer()
        self.conv3 = bottleneck.conv3
        self.bn3 = bottleneck.bn3
        self.activ = bottleneck.activ
        self.downsample = bottleneck.downsample
        self.stride = bottleneck.stride
        
    def forward(self, x):
        residual = x
        x = self.quant1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activ(out)
        out = self.quant2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activ(out)
        out = self.quant3(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activ(out)

        return out


supported_bitsplit = {
    BasicBlock : QBasicBlock,
    Bottleneck : QBottleneck
}

class QuantModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        setattr(model, 'quant', Quantizer())
        setattr(model, 'fc', nn.Sequential(model.quant, model.fc))
        self.quantizer(model)

    def quantizer(self, module: nn.Module):
        for name, child_module in module.named_children():
            if type(child_module) in supported_bitsplit:
                setattr(module, name, supported_bitsplit[type(child_module)](child_module))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.ReLU6)):
                continue
            else: self.quantizer(child_module)

