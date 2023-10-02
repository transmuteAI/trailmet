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
from typing import Union, Dict, Callable
from trailmet.models.resnet import BasicBlock, Bottleneck
from trailmet.models.mobilenet import InvertedResidual
from trailmet.algorithms.quantize.methods import UniformAffineQuantizer, ActQuantizer
from trailmet.algorithms.quantize._methods import BaseQuantizer, UniformQuantizer, AdaRoundQuantizer
from trailmet.algorithms.quantize.utils import get_qscheme, get_dtype
from torch.ao.quantization import QConfig, FixedQParamsObserver
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic as nni

__all__ = [
    'StraightThrough',
    'QuantModule',
    'BaseQuantBlock',
    'QuantBasicblock',
    'QuantBottleneck',
    'QuantInvertedResidual',
    # old modules kept temporarily for BC
    'QBasicblock'
    'QBottleneck',
    'QInvertedResidual',
]

QUANTIZER_MAPPING: Dict[str, Callable] = {
    'uniform': UniformQuantizer,
    'adaround': AdaRoundQuantizer
}
class StraightThrough(nn.Module):
    """
    Identity Layer, same as torch.nn.modules.linear.Identity
    """
    def __int__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input

class QuantModule(nn.Module):
    """
    Wrapper Module to simulate fake quantization
    """
    def __init__(self, orig_module: nn.Module, weight_qparams: dict, act_qparams: dict):
        super().__init__()
        self.orig_module = orig_module
        if isinstance(orig_module, (nni.ConvReLU2d, nni.LinearReLU)):
            assert len(orig_module)==2
            if type(orig_module[0]) == nn.Conv2d:
                self.fwd_func = F.conv2d
                self.fwd_kwargs = dict(
                    stride = orig_module[0].stride,
                    padding = orig_module[0].padding,
                    dilation = orig_module[0].dilation,
                    groups = orig_module[0].groups
                )
            elif type(orig_module[0]) == nn.Linear:
                self.fwd_func = F.linear
                self.fwd_kwargs = dict()
            else: 
                raise NotImplementedError
            
            if type(orig_module[1]) == nn.ReLU:
                self.fwd_post = F.relu
            else:
                raise NotImplementedError
            
            self.weight = orig_module[0].weight
            self.orig_weight = orig_module[0].weight.data.clone()
            self.bias = orig_module[0].bias

        elif isinstance(orig_module, (nn.Conv2d, nn.Linear)):
            if type(orig_module) == nn.Conv2d:
                self.fwd_func = F.conv2d
                self.fwd_kwargs = dict(
                    stride = orig_module.stride,
                    padding = orig_module.padding,
                    dilation = orig_module.dilation,
                    groups = orig_module.groups
                )
            elif type(orig_module) == nn.Linear:
                self.fwd_func = F.linear
                self.fwd_kwargs = dict()
            else:
                raise NotImplementedError
            
            self.fwd_post = self.identity
            self.weight = orig_module.weight
            self.orig_weight = orig_module.weight.data.clone()
            self.bias = orig_module.bias

        else:
            raise NotImplementedError

        self.weight_quantizer: BaseQuantizer = QUANTIZER_MAPPING[weight_qparams.get(
            'quantizer', 'uniform')](weight_qparams)
        self.act_quantizer: BaseQuantizer = QUANTIZER_MAPPING[act_qparams.get(
            'quantizer', 'uniform')](act_qparams)

        self.use_act_quant = False
        self.use_weight_quant = False
        self.ignore_reconstruction = False
        self.extra_repr = orig_module.extra_repr


    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.orig_weight
            bias = self.bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.fwd_post(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out 


    def identity(self, x: torch.Tensor):
        return x

    def set_observation_state(self, weight_obs: bool, act_obs: bool):
        self.weight_quantizer.enable_observation = weight_obs
        self.act_quantizer.enable_observation = act_obs

    def set_quantization_state(self, weight_quant: bool, act_quant: bool):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant



class BaseQuantBlock(nn.Module):
    def __init__(self, act_qparams: dict) -> None:
        super().__init__()
        self.act_quantizer: BaseQuantizer = QUANTIZER_MAPPING[act_qparams.get(
            'quantizer', 'uniform')](act_qparams)
        self.use_act_quant = False
        self._fake_quantization = True
        self.ignore_reconstruction = False

    def set_observation_state(self, weight_obs: bool, act_obs: bool):
        self.act_quantizer.enable_observation = act_obs
        for module in self.modules():
            if isinstance(module, QuantModule):
                module.set_observation_state(weight_obs, act_obs)

    def set_quantization_state(self, weight_quant: bool, act_quant: bool):
        self.use_act_quant = act_quant
        for module in self.modules():
            if isinstance(module, QuantModule):
                module.set_quantization_state(weight_quant, act_quant)


    def _convert_to_quantizable_with_qconfig(self, module: nn.Module):
        self._fake_quantization = False
        module_attach = dict()
        module_reassign = dict()
        
        for name, submodule in module.named_modules():
            if isinstance(submodule, QuantModule):
                module_attach[name]['weight'] = submodule.weight_quantizer.observer
                module_attach[name]['activation'] = submodule.act_quantizer.observer
                module_reassign[name] = submodule.orig_module
        
        for name, orig_module in module_reassign.items():  
            delattr(module, name)
            setattr(module, name, orig_module)
        
        for name, observers in module_attach.items():
            submodule = getattr(module, name, None)
            assert submodule is not None
            if isinstance(submodule, nni.ConvReLU2d):   # propagate qconfig
                setattr(submodule[0], 'qconfig', QConfig(
                    weight = observers['weight'],
                    activation = None
                ))
            setattr(submodule, 'qconfig', QConfig(
                weight = observers['weight'],
                activation = observers['activation']
            ))
            submodule.add_module('activation_post_process', submodule.qconfig.activation())                    



class QuantBasicBlock(BaseQuantBlock):
    def __init__(self, basicblock: BasicBlock, weight_qparams: dict, act_qparams: dict):
        super().__init__(act_qparams)
        # assuming all bn and relu are fused in conv
        self.conv1 = QuantModule(basicblock.conv1, weight_qparams, act_qparams)
        self.conv2 = QuantModule(basicblock.conv2, weight_qparams, act_qparams)
        if basicblock.downsample is not None:
            self.downsample = QuantModule(basicblock.downsample[0], weight_qparams, act_qparams)
        else:
            self.downsample = None
        self.add_skip = nnq.FloatFunctional()


    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        skip = inp if self.downsample is None else self.downsample(inp)
        out = self.conv1(inp)
        out = self.conv2(out)
        out = self.add_skip.add_relu(out, skip)
        if self._fake_quantization and self.use_act_quant:
            out = self.act_quantizer(out)
        return out
    

    def _convert_to_quantizable_with_qconfig(self):
        super()._convert_to_quantizable_with_qconfig(self)
        setattr(self.add_skip, 'qconfig', QConfig(
            weight = None,
            activation = self.act_quantizer.observer
        ))
        self.add_skip.add_module('activation_post_process', 
            self.add_skip.qconfig.activation())



class QuantBottleneck(BaseQuantBlock):
    def __init__(self, bottleneck: Bottleneck, weight_qparams: dict, act_qparams: dict) -> None:
        super().__init__(act_qparams)
        # assuming all bn and relu are fused in conv
        self.conv1 = QuantModule(bottleneck.conv1, weight_qparams, act_qparams)     # ConvReLU2d
        self.conv2 = QuantModule(bottleneck.conv2, weight_qparams, act_qparams)     # ConvReLU2d
        self.conv3 = QuantModule(bottleneck.conv3, weight_qparams, act_qparams)     # ConvReLU2d
        if bottleneck.downsample is not None:
            self.downsample = QuantModule(bottleneck.downsample[0], weight_qparams, act_qparams)   # Conv2d
        else:
            self.downsample = None
        self.add_skip = nnq.FloatFunctional()
        

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        skip = inp if self.downsample is None else self.downsample(inp)
        out = self.conv1(inp)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.add_skip.add_relu(out, skip)
        if self._fake_quantization and self.use_act_quant:
            out = self.act_quantizer(out)
        return out


    def _convert_to_quantizable_with_qconfig(self):
        super()._convert_to_quantizable_with_qconfig(self)
        setattr(self.add_skip, 'qconfig', QConfig(
            weight = None,
            activation = self.act_quantizer.observer
        ))
        self.add_skip.add_module('activation_post_process', 
            self.add_skip.qconfig.activation())



class QuantInvertedResidual(BaseQuantBlock):
    def __init__(self, act_quant_params: dict = {}) -> None:
        super().__init__(act_quant_params)





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

    
        