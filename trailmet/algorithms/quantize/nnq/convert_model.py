import copy
from typing import Dict, Callable, Any
from torch import nn as nn
from torch.nn.utils.parametrize import type_before_parametrizations as _type
from torch.ao.nn import quantized as nnq
from torch.ao.nn import intrinsic as nni
from torch.ao.nn.intrinsic import quantized as nniq
from torch.ao.nn.intrinsic.modules.fused import _FusedModule
from torch.ao.quantization.stubs import QuantStub, DeQuantStub

DEFAULT_STATIC_QUANT_MODULE_MAPPING: Dict[Callable, Callable] = {
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    nn.BatchNorm2d: nnq.BatchNorm2d,
    nn.Dropout: nnq.Dropout,
    nn.Conv2d: nnq.Conv2d,
    nn.Linear: nnq.Linear,
    nn.ReLU6: nnq.ReLU6,
    nn.LeakyReLU: nnq.LeakyReLU,
    # Wrapper Modules:
    nnq.FloatFunctional: nnq.QFunctional,
    # Intrinsic modules:
    nni.BNReLU2d: nniq.BNReLU2d,
    nni.ConvReLU2d: nniq.ConvReLU2d,
    nni.ConvAdd2d: nniq.ConvAdd2d,
    nni.ConvAddReLU2d: nniq.ConvAddReLU2d,
    nni.LinearReLU: nniq.LinearReLU,
    nni.LinearLeakyReLU: nniq.LinearLeakyReLU,
}   

def get_qparam_dict(observer):
    qparams = dict()
    qparams["qscheme"] = getattr(observer, "qscheme", None)
    qparams["dtype"] = observer.dtype
    qparams["scale"], qparams["zero_point"] = observer.calculate_qparams()
    if hasattr(observer, "quant_min"):
        qparams["quant_min"] = observer.quant_min
    if hasattr(observer, "quant_max"):
        qparams["quant_max"] = observer.quant_max
    return qparams


def _observer_forward_hook(self, input, output):
    """forward hook that calls observer on the output"""
    return self.activation_post_process(output)

def _observer_forward_pre_hook(self, input):
    r"""Forward pre hook that calls observer on the input
    """
    return self.activation_post_process(input[0])


def _remove_qconfig(module):
    for child_module in module.children():
        _remove_qconfig(child_module)
    
    if hasattr(module, 'qconfig'):
        del module.qconfig
    
    if hasattr(module, 'activation_post_process'):
        del module.activation_post_process



def convert_model(model, inplace=True, remove_qconfig=True):
    if not inplace:
        model = copy.deepcopy(model)
    mapping = DEFAULT_STATIC_QUANT_MODULE_MAPPING
    _convert(model, mapping, inplace)
    if remove_qconfig:
        _remove_qconfig(model)
    return model

def _replace_relu(module: nn.Module) -> None:
    """replace all ReLU6 with ReLU"""
    reassign = {}
    for name, child_module in module.named_children():
        _replace_relu(child_module)
        if type(child_module) is nn.ReLU or type(child_module) is nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value

def _convert(module: nn.Module, mapping, inplace=True):
    """recursively convert modules to their quantized counterparts"""
    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}
    for name, child_module in module.named_children():
        if not isinstance(child_module, _FusedModule):  # fused modules are swapped as one unit
            _convert(child_module, mapping, True)
        reassign[name] = swap_module(child_module, mapping)

    for name, quantized_module in reassign.items():
        module._modules[name] = quantized_module
    
    return module


def swap_module(module, mapping):
    """
    swaps the module if it has a quantized counterpart and 
    if it has an `observer` attached.
    Args: 
        module: input module
        mapping: a dict that maps from nn/nni module to nnq/nniq module 
    Return:
        corresponding quantized counterpart of `module`
    """
    new_module = module
    
    if hasattr(module, 'qconfig') and module.qconfig is not None:
        swapped = False
        if _type(module) in mapping:
            qmod = mapping[_type(module)]
            new_module = qmod.from_float(module)
            swapped = True
    
    if swapped:
        # Preserve module's pre forward hooks. They'll be called on quantized input
        for pre_hook_fn in module._forward_pre_hooks.values():
            new_module.register_forward_pre_hook(pre_hook_fn)
        # Preserve module's post forward hooks except _observer_forward_hook
        # After convert they'll work with quantized output
        for hook_fn in module._forward_hooks.values():
            if hook_fn is not _observer_forward_hook:
                new_module.register_forward_hook(hook_fn)
    
    return new_module
