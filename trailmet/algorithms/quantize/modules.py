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
import torch.nn.functional as F
from typing import Union
from trailmet.models.resnet import BasicBlock, Bottleneck
from trailmet.models.mobilenet import InvertedResidual
from trailmet.algorithms.quantize.methods import UniformAffineQuantizer, ActQuantizer


#============================================
#***** Quantization Modules for BRECQ *******
#============================================
"""
Supported quantization wrappers for pytorch modules :-
    - BasicBlock(nn.Module) -> QuantBasicBlock(BaseQuantBlock(nn.Module))
    - Bottleneck(nn.Module) -> QuantBottleneck(BaseQuantBlock(nn.Module))
    - InvertedResidual(nn.Module) -> QuantInvertedResidual(BaseQuantBlock(nn.Module))
        - nn.Conv2d, nn.Linear -> QuantModule(nn.Module)
"""
class StraightThrough(nn.Module):
    """Identity Layer"""
    def __int__(self):
        super().__init__()
        pass

    def forward(self, input):
        return input

class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
            act_quant_params: dict = {}, disable_act_quant: bool = False, se_module = None):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
            
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        
        # initialize quantizer
        self.weight_quantizer = weight_quant_params.get('method', UniformAffineQuantizer)(**weight_quant_params)
        self.act_quantizer = act_quant_params.get('method', UniformAffineQuantizer)(**act_quant_params)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.se_module = se_module
        self.extra_repr = org_module.extra_repr

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        if self.se_module is not None:
            out = self.se_module(out)
        out = self.activation_function(out)
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant


class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """
    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        self.act_quantizer = act_quant_params.get('method', UniformAffineQuantizer)(**act_quant_params)
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)

class QuantBasicBlock(BaseQuantBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, basic_block: BasicBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = basic_block.activ
        self.conv2 = QuantModule(basic_block.conv2, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.activation_function = basic_block.activ

        if basic_block.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(basic_block.downsample[0], weight_quant_params, act_quant_params,
                disable_act_quant=True)
        # copying all attributes in original block
        self.stride = basic_block.stride

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

class QuantBottleneck(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """

    def __init__(self, bottleneck: Bottleneck, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = bottleneck.activ
        self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params, act_quant_params)
        self.conv2.activation_function = bottleneck.activ
        self.conv3 = QuantModule(bottleneck.conv3, weight_quant_params, act_quant_params, disable_act_quant=True)

        # modify the activation function to ReLU
        self.activation_function = bottleneck.activ

        if bottleneck.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(bottleneck.downsample[0], weight_quant_params, act_quant_params,
                disable_act_quant=True)
        # copying all attributes in original block
        self.stride = bottleneck.stride

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

class QuantInvertedResidual(BaseQuantBlock):
    """
    Implementation of Quantized Inverted Residual Block used in MobileNetV2.
    Inverted Residual does not have activation function.
    """

    def __init__(self, inv_res: InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.stride = inv_res.stride
        self.inp = inv_res.inp
        self.oup = inv_res.oup
        self.exp = inv_res.exp
        self.conv1 = QuantModule(inv_res.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = nn.ReLU6(inplace=True)
        self.conv2 = QuantModule(inv_res.conv2, weight_quant_params, act_quant_params)
        self.conv2.activation_function = nn.ReLU6(inplace=True)
        self.conv3 = QuantModule(inv_res.conv3, weight_quant_params, act_quant_params)
        self.shortcut = nn.Sequential()
        if self.stride==1 and self.inp!=self.oup:
            self.shortcut = nn.Sequential(
                QuantModule(inv_res.shortcut[0], weight_quant_params, act_quant_params)
            )
        # self.use_res_connect = inv_res.use_res_connect
        # self.expand_ratio = inv_res.exp
        # if self.expand_ratio == 1:
        #     self.conv = nn.Sequential(
        #         QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
        #         QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params, disable_act_quant=True),
        #     )
        #     self.conv[0].activation_function = nn.ReLU6()
        # else:
        #     self.conv = nn.Sequential(
        #         QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
        #         QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params),
        #         QuantModule(inv_res.conv[6], weight_quant_params, act_quant_params, disable_act_quant=True),
        #     )
        #     self.conv[0].activation_function = nn.ReLU6()
        #     self.conv[1].activation_function = nn.ReLU6()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out
        # if self.use_res_connect:
        #     out = x + self.conv(x)
        # else:
        #     out = self.conv(x)
        # out = self.activation_function(out)
        # if self.use_act_quant:
        #     out = self.act_quantizer(out)
        # return out


#===============================================
#***** Quantization Modules for BitSplit *******
#===============================================
"""
Supported quantization wrappers for pytorch modules :-
    - BasicBlock(nn.Module) -> QBasicBlock(nn.Module)
    - Bottleneck(nn.Module) -> QBottleneck(nn.Module)
    - InvertedResidual(nn.Module) -> QInvertedResidual(nn.Module)
"""

class QBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, basic_block: BasicBlock):
        super().__init__()
        self.quant1 = ActQuantizer()
        self.conv1 = basic_block.conv1
        self.bn1 = basic_block.bn1
        self.activ = basic_block.activ
        self.quant2 = ActQuantizer()
        self.conv2 = basic_block.conv2
        self.bn2 = basic_block.bn2
        self.downsample = basic_block.downsample
        self.stride = basic_block.stride
        
    def forward(self, x):
        residual = x
        x = self.quant1(x)
        out = self.activ(self.bn1(self.conv1(x)))
        out = self.quant2(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activ(out)
        return out


class QBottleneck(nn.Module):
    expansion = 4
    def __init__(self, bottleneck: Bottleneck):
        super().__init__()
        self.quant1 = ActQuantizer()
        self.conv1 = bottleneck.conv1
        self.bn1 = bottleneck.bn1
        self.quant2 = ActQuantizer()
        self.conv2 = bottleneck.conv2
        self.bn2 = bottleneck.bn2
        self.quant3 = ActQuantizer()
        self.conv3 = bottleneck.conv3
        self.bn3 = bottleneck.bn3
        self.activ = bottleneck.activ
        self.downsample = bottleneck.downsample
        self.stride = bottleneck.stride
        
    def forward(self, x):
        residual = x
        x = self.quant1(x)
        out = self.activ(self.bn1(self.conv1(x)))
        out = self.quant2(out)
        out = self.activ(self.bn2(self.conv2(out)))
        out = self.quant3(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activ(out)
        return out


class QInvertedResidual(nn.Module):
    def __init__(self, inv_res: InvertedResidual):
        super().__init__() 
        self.stride = inv_res.stride
        self.inp = inv_res.inp
        self.oup = inv_res.oup
        self.exp = inv_res.exp
        self.quant1 = ActQuantizer(islinear=1)
        self.conv1 = inv_res.conv1
        self.bn1 = inv_res.bn1
        self.quant2 = ActQuantizer(islinear=1)
        self.conv2 = inv_res.conv2
        self.bn2 = inv_res.bn2
        self.quant3 = ActQuantizer(islinear=0)
        self.conv3 = inv_res.conv3
        self.bn3 = inv_res.bn3
        self.shortcut = inv_res.shortcut

    def forward(self, x):
        x = self.quant1(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.quant2(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.quant3(out)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

    
        