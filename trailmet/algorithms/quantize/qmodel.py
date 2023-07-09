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
from trailmet.algorithms.quantize.quantize import StraightThrough
from trailmet.algorithms.quantize.methods import (
    MaxAbsStaticQuantization,
    LpNormQuantization,
)
from trailmet.algorithms.quantize.methods import UniformAffineQuantizer
from trailmet.algorithms.quantize.methods import ActQuantizer

__all__ = [
    'QuantBasicBlock',
    'QuantBottleneck',
    'QuantInvertedResidual',
    'QuantModule',
    'BaseQuantBlock',
    'QBasicBlock',
    'QBottleneck',
    'QInvertedResidual',
    'ActivationModuleWrapper',
    'ParameterModuleWrapper',
]
# ============================================
# ***** Quantization Modules for BRECQ *******
# ============================================
"""
Supported quantization wrappers for pytorch modules :-
    - BasicBlock(nn.Module) -> QuantBasicBlock(BaseQuantBlock(nn.Module))
    - Bottleneck(nn.Module) -> QuantBottleneck(BaseQuantBlock(nn.Module))
    - InvertedResidual(nn.Module) -> QuantInvertedResidual(BaseQuantBlock(nn.Module))
        - nn.Conv2d, nn.Linear -> QuantModule(nn.Module)
"""


class QuantModule(nn.Module):
    """Quantized Module that can perform quantized convolution or normal
    convolution.

    To activate quantization, please use set_quant_state function.

    Parameters
    ----------
    org_module (nn.Module): Module to be used
    weight_quant_params (dict): Weight parameters
    act_quant_params (dict): Activation Parameters
    disable_act_quant (bool): if True, activation layer will be disabled
    se_module (nn.Module): SE Module to be used
    """

    def __init__(
        self,
        org_module: Union[nn.Conv2d, nn.Linear],
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_act_quant: bool = False,
        se_module=None,
    ):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(
                stride=org_module.stride,
                padding=org_module.padding,
                dilation=org_module.dilation,
                groups=org_module.groups,
            )
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
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

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

    def set_quant_state(self,
                        weight_quant: bool = False,
                        act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant


class BaseQuantBlock(nn.Module):
    """Base implementation of block structures for all networks.

    Due to the branch architecture, we have to perform activation function and
    quantization after the elemental-wise add operation, therefore, we put this
    part in this class.

    Parameters
    ----------
    act_quant_params (dict): Activation parameters
    """

    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

    def set_quant_state(self,
                        weight_quant: bool = False,
                        act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


class QuantBasicBlock(BaseQuantBlock):
    """Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.

    Parameters
    ----------
    basic_block (object): BasicBlock which is to be used
    weight_quant_params (dict): Weight parameters
    act_quant_params (dict): Activation Parameters
    """

    def __init__(
        self,
        basic_block: BasicBlock,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
    ):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params,
                                 act_quant_params)
        self.conv1.activation_function = basic_block.active
        self.conv2 = QuantModule(
            basic_block.conv2,
            weight_quant_params,
            act_quant_params,
            disable_act_quant=True,
        )

        # modify the activation function to ReLU
        self.activation_function = basic_block.active

        if basic_block.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(
                basic_block.downsample[0],
                weight_quant_params,
                act_quant_params,
                disable_act_quant=True,
            )
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
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and
    -152.

    Parameters
    ----------
    bottleneck (object): Bottleneck to be used
    weight_quant_params (dict): Weight parameters
    act_quant_params (dict): Activation Parameters
    """

    def __init__(
        self,
        bottleneck: Bottleneck,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
    ):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params,
                                 act_quant_params)
        self.conv1.activation_function = bottleneck.active
        self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params,
                                 act_quant_params)
        self.conv2.activation_function = bottleneck.active
        self.conv3 = QuantModule(
            bottleneck.conv3,
            weight_quant_params,
            act_quant_params,
            disable_act_quant=True,
        )

        # modify the activation function to ReLU
        self.activation_function = bottleneck.active

        if bottleneck.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(
                bottleneck.downsample[0],
                weight_quant_params,
                act_quant_params,
                disable_act_quant=True,
            )
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
    """Implementation of Quantized Inverted Residual Block used in MobileNetV2.

    Inverted Residual does not have activation function.

    Parameters
    ----------
    inv_res (object): Inverted Residual block to be used
    weight_quant_params (dict): Weight parameters
    act_quant_params (dict): Activation Parameters
    """

    def __init__(
        self,
        inv_res: InvertedResidual,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
    ):
        super().__init__(act_quant_params)
        self.stride = inv_res.stride
        self.inp = inv_res.inp
        self.oup = inv_res.oup
        self.exp = inv_res.exp
        self.conv1 = QuantModule(inv_res.conv1, weight_quant_params,
                                 act_quant_params)
        self.conv1.activation_function = nn.ReLU6(inplace=True)
        self.conv2 = QuantModule(inv_res.conv2, weight_quant_params,
                                 act_quant_params)
        self.conv2.activation_function = nn.ReLU6(inplace=True)
        self.conv3 = QuantModule(inv_res.conv3, weight_quant_params,
                                 act_quant_params)
        self.shortcut = nn.Sequential()
        if self.stride == 1 and self.inp != self.oup:
            self.shortcut = nn.Sequential(
                QuantModule(inv_res.shortcut[0], weight_quant_params,
                            act_quant_params))
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
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out
        # if self.use_res_connect:
        #     out = x + self.conv(x)
        # else:
        #     out = self.conv(x)
        # out = self.activation_function(out)
        # if self.use_act_quant:
        #     out = self.act_quantizer(out)
        # return out


# ===============================================
# ***** Quantization Modules for BitSplit *******
# ===============================================
"""
Supported quantization wrappers for pytorch modules :-
    - BasicBlock(nn.Module) -> QBasicBlock(nn.Module)
    - Bottleneck(nn.Module) -> QBottleneck(nn.Module)
    - InvertedResidual(nn.Module) -> QInvertedResidual(nn.Module)
"""


class QBasicBlock(nn.Module):
    """
    Parameters
    ----------
    basic_block (object): BasicBlock which is to be used
    """

    expansion = 1

    def __init__(self, basic_block: BasicBlock):
        super().__init__()
        self.quant1 = ActQuantizer()
        self.conv1 = basic_block.conv1
        self.bn1 = basic_block.bn1
        self.active = basic_block.active
        self.quant2 = ActQuantizer()
        self.conv2 = basic_block.conv2
        self.bn2 = basic_block.bn2
        self.downsample = basic_block.downsample
        self.stride = basic_block.stride

    def forward(self, x):
        residual = x
        x = self.quant1(x)
        out = self.active(self.bn1(self.conv1(x)))
        out = self.quant2(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.active(out)
        return out


class QBottleneck(nn.Module):
    """
    Parameters
    ----------
    bottleneck (object): Bottleneck to be used
    """

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
        self.active = bottleneck.active
        self.downsample = bottleneck.downsample
        self.stride = bottleneck.stride

    def forward(self, x):
        residual = x
        x = self.quant1(x)
        out = self.active(self.bn1(self.conv1(x)))
        out = self.quant2(out)
        out = self.active(self.bn2(self.conv2(out)))
        out = self.quant3(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.active(out)
        return out


class QInvertedResidual(nn.Module):
    """
    Parameters
    ----------
    inv_res (object): Inverted Residual block to be used
    """

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
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


# ===========================================
# ***** Quantization Modules for LAPQ *******
# ===========================================
"""
Supported quantization wrappers for pytorch modules :-
    - nn.ReLU, nn.ReLU6 -> ActivationModuleWrapper(nn.Module)
    - nn.Conv2d, nn.Linear -> ParameterModuleWrapper(nn.Module)
"""

quantization_mapping = {
    'max_static': MaxAbsStaticQuantization,
    'lp_norm': LpNormQuantization,
}


def is_positive(module):
    return isinstance(module, nn.ReLU) or isinstance(module, nn.ReLU6)


class ActivationModuleWrapper(nn.Module):
    """
    Parameters
    ----------
    name (str): Name of the wrapped module
    wrapped_module (object): Module to be used
    kwargs (object): A yaml safe loaded file with information like bits_out and qtype
    """

    def __init__(self, name, wrapped_module, **kwargs):
        super(ActivationModuleWrapper, self).__init__()
        self.name = name
        self.wrapped_module = wrapped_module
        self.bits_out = kwargs['bits_out']
        self.qtype = kwargs['qtype']
        self.post_relu = True
        self.enabled = True
        self.active = True
        if self.bits_out is not None:
            self.out_quantization = self.out_quantization_default = None

            def __init_out_quantization__(tensor):
                self.out_quantization_default = quantization_mapping[
                    self.qtype](
                        self,
                        tensor,
                        self.bits_out,
                        symmetric=(not is_positive(wrapped_module)),
                        uint=True,
                        kwargs=kwargs,
                    )
                self.out_quantization = self.out_quantization_default

            self.out_quantization_init_fn = __init_out_quantization__

    def __enabled__(self):
        return self.enabled and self.active and self.bits_out is not None

    def forward(self, *input):
        if self.post_relu:
            out = self.wrapped_module(*input)
            # Quantize output
            if self.__enabled__():
                self.verify_initialized(self.out_quantization, out,
                                        self.out_quantization_init_fn)
                out = self.out_quantization(out)
        else:
            # Quantize output
            if self.__enabled__():
                self.verify_initialized(self.out_quantization, *input,
                                        self.out_quantization_init_fn)
                out = self.out_quantization(*input)
            else:
                out = self.wrapped_module(*input)
        return out

    @staticmethod
    def verify_initialized(quantization_handle, tensor, init_fn):
        if quantization_handle is None:
            init_fn(tensor)

    def get_quantization(self):
        return self.out_quantization

    def set_quantization(self, qtype, kwargs):
        self.out_quantization = qtype(
            self,
            self.bits_out,
            symmetric=(not is_positive(self.wrapped_module)),
            uint=True,
            kwargs=kwargs,
        )


class ParameterModuleWrapper(nn.Module):
    """
    Parameters
    ----------
    name (str): Name of the wrapped module
    wrapped_module (object): Module to be used
    kwargs (object): A yaml safe loaded file with information like bits_out, qtype, forward functor, bit_weight, etc.
    """

    def __init__(self, name, wrapped_module, **kwargs):
        super(ParameterModuleWrapper, self).__init__()
        self.name = name
        self.wrapped_module = wrapped_module
        self.forward_functor = kwargs['forward_functor']
        self.bit_weights = kwargs['bit_weights']
        self.bits_out = kwargs['bits_out']
        self.qtype = kwargs['qtype']
        self.bcorr_w = kwargs['bcorr_w']
        self.bn = kwargs['bn'] if 'bn' in kwargs else None
        self.enabled = True
        self.active = True
        self.centroids_hist = {}
        self.log_weights_hist = False
        self.log_weights_mse = False
        self.log_clustering = False
        self.dynamic_weight_quantization = True
        setattr(self, 'weight', wrapped_module.weight)
        delattr(wrapped_module, 'weight')
        if hasattr(wrapped_module, 'bias'):
            setattr(self, 'bias', wrapped_module.bias)
            delattr(wrapped_module, 'bias')
        if self.bit_weights is not None:
            self.weight_quantization_default = quantization_mapping[
                self.qtype](
                    self,
                    self.weight,
                    self.bit_weights,
                    symmetric=True,
                    uint=True,
                    kwargs=kwargs,
                )
            self.weight_quantization = self.weight_quantization_default
            if not self.dynamic_weight_quantization:
                self.weight_q = self.weight_quantization(self.weight)
                self.weight_mse = torch.mean(
                    (self.weight_q - self.weight)**2).item()

    def __enabled__(self):
        return self.enabled and self.active and self.bit_weights is not None

    def bias_corr(self, x, xq):
        bias_q = xq.view(xq.shape[0], -1).mean(-1)
        bias_orig = x.view(x.shape[0], -1).mean(-1)
        bcorr = bias_q - bias_orig
        return (xq - bcorr.view(bcorr.numel(), 1, 1, 1)
                if len(x.shape) == 4 else xq - bcorr.view(bcorr.numel(), 1))

    def forward(self, *input):
        w = self.weight
        if self.__enabled__():
            # Quantize weights
            if self.dynamic_weight_quantization:
                w = self.weight_quantization(self.weight)
                if self.bcorr_w:
                    w = self.bias_corr(self.weight, w)
            else:
                w = self.weight_q
        out = self.forward_functor(
            *input,
            weight=w,
            bias=(self.bias if hasattr(self, 'bias') else None))
        return out

    def get_quantization(self):
        return self.weight_quantization

    def set_quantization(self, qtype, kwargs):
        self.weight_quantization = qtype(self,
                                         self.bit_weights,
                                         symmetric=True,
                                         uint=True,
                                         kwargs=kwargs)
