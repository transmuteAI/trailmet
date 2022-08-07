# wrapper classes for building quantization modules
# source: https://github.com/yhhhli/BRECQ/tree/main/quant

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy
from typing import Union
from trailmet.models.resnet import BasicBlock, Bottleneck
from trailmet.models.mobilenetv2 import InvertedResidual
from trailmet.algorithms.quantize.quantize import StraightThrough, FoldBN
from trailmet.algorithms.quantize.methods import UniformAffineQuantizer, LpNormQuantizer, FixQuantizationClipValue


quantization_mapping = {
    'uaq' : UniformAffineQuantizer,
    'lp_norm' : LpNormQuantizer,
    'fix_clip' : FixQuantizationClipValue
}

#=========================
##### Quantize Layer #####
#=========================

class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, orig_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, se_module=None):
        super(QuantModule, self).__init__()
        if isinstance(orig_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=orig_module.stride, padding=orig_module.padding,
                                   dilation=orig_module.dilation, groups=orig_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = orig_module.weight
        self.orig_weight = orig_module.weight.data.clone()
        if orig_module.bias is not None:
            self.bias = orig_module.bias
            self.orig_bias = orig_module.bias.data.clone()
        else:
            self.bias = None
            self.orig_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        self.activation_function = StraightThrough()
        # initialize quantizer
        assert weight_quant_params.get('qtype', 'uaq') == act_quant_params.get('qtype', 'uaq'), 'qtype mismatch'
        self.qtype = weight_quant_params.get('qtype', 'uaq')
        self.bcorr = False if self.qtype=='uaq' else True
        if self.qtype=='uaq':
            self.bcorr = False
            self.weight_quantizer = quantization_mapping[self.qtype](**weight_quant_params)
            self.act_quantizer = quantization_mapping[self.qtype](**act_quant_params)
        else:
            self.bcorr = True
            self.weight_quantizer = quantization_mapping[self.qtype](self, self.weight, **weight_quant_params) # must pass weight here
            self.act_quantizer = self.act_quantization_default = None
            def __init_out_quantization__(tensor):
                self.act_quantization_default = quantization_mapping[self.qtype](self, tensor, **act_quant_params) # To Do: make it assymmetric for ReLU
                self.act_quantizer = self.act_quantization_default
            self.act_quantization_init_fn = __init_out_quantization__

        self.ignore_reconstruction = False
        self.se_module = se_module
        self.extra_repr = orig_module.extra_repr

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
            if self.bcorr:
                weight = self.bias_corr(self.weight, weight)
        else:
            weight = self.orig_weight
            bias = self.orig_bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        if self.se_module is not None:
            out = self.se_module(out)
        out = self.activation_function(out)
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            if self.qtype!='uaq':
                self.verify_initialized(self.act_quantizer, out, self.act_quantization_init_fn)
            out = self.act_quantizer(out)
        return out

    def bias_corr(self, x: torch.Tensor, xq: torch.Tensor):
        bias_q = xq.view(xq.shape[0], -1).mean(-1)
        bias_orig = x.view(x.shape[0], -1).mean(-1)
        bcorr = bias_q - bias_orig

        return xq - bcorr.view(bcorr.numel(), 1, 1, 1) if len(x.shape) == 4 else xq - bcorr.view(bcorr.numel(), 1)

    @staticmethod
    def verify_initialized(quantization_handle, tensor, init_fn):
        if quantization_handle is None:
            init_fn(tensor)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
    
    def get_quantization(self):
        return self.weight_quantizer

    def set_quantization(self, clip_value=None, device=None, new_qtype='fix_clip', **kwargs):
        assert self.qtype != 'uaq', 'quant type not supported by LAPQ'
        self.weight_quantizer = quantization_mapping[new_qtype](self, clip_value=clip_value, device=device, **kwargs)


#=========================
##### Quantize Block #####
#=========================

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
        # initialize quantizer

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
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

        # modify the activation function to ReLU
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

        self.use_res_connect = inv_res.use_res_connect
        self.expand_ratio = inv_res.expand_ratio
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].activation_function = nn.ReLU6()
        else:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[6], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].activation_function = nn.ReLU6()

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


replacement_factory = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    InvertedResidual: QuantInvertedResidual,
}

#=========================
##### Quantize Model #####
#=========================

class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.model = copy.deepcopy(model)
        bn = FoldBN()
        bn.search_fold_and_remove_bn(self.model)
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if type(child_module) in replacement_factory:
                setattr(module, name, replacement_factory[type(child_module)](child_module, weight_quant_params, act_quant_params))

            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, input):
        return self.model(input)

    def set_first_last_layer_to_8bit(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[0].weight_quantizer.bitwidth_refactor(8)
        module_list[0].act_quantizer.bitwidth_refactor(8)
        module_list[-1].weight_quantizer.bitwidth_refactor(8)
        module_list[-2].act_quantizer.bitwidth_refactor(8)
        # ignore reconstruction of the first layer
        module_list[0].ignore_reconstruction = True

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True

    def synchorize_activation_statistics(self):
        # import linklink.dist_helper as dist
        for m in self.modules():
            if isinstance(m, QuantModule):
                if m.act_quantizer.delta is not None:
                    m.act_quantizer.delta.data /= dist.get_world_size()
                    dist.all_reduce(m.act_quantizer.delta.data)

    def set_quant_params(self, scales: list, device=None, **kwargs):
        i=0
        for module in self.model.modules():
            if isinstance(module, QuantModule):
                module.set_quantization(
                    clip_value=scales[i], device=device, new_qtype='fix_clip', **kwargs)
                i+=1
        
    def get_quant_params(self):
        scales = []
        for module in self.model.modules():
            if isinstance(module, QuantModule):
                q = module.get_quantization()
                assert hasattr(q, 'alpha'), 'quant module has no atttribute alpha'
                clip_value = getattr(q, 'alpha')
                scales.append(clip_value.item())
        return scales