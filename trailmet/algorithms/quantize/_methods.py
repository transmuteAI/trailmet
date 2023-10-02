import torch
import torch.nn as nn
from typing import Dict, Callable
from trailmet.algorithms.quantize.observers import BaseObserver, MinMaxObserver, LpNormObserver
from trailmet.algorithms.quantize.utils import reshape_qparams_by_channel



OBSERVER_MAPPING: Dict[str, Callable] = {
    'min_max': MinMaxObserver,
    'lp_norm': LpNormObserver
}


class RoundSTE(torch.autograd.Function):
    """grad enabled round function"""
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class FloorSTE(torch.autograd.Function):
    """grad enabled floor function"""
    @staticmethod
    def forward(ctx, input):
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BaseQuantizer(nn.Module):
    def __init__(self, kwargs: dict):
        self.observer: BaseObserver = OBSERVER_MAPPING[kwargs.get(
            'observer', 'min_max')](**kwargs)
        self.quant_min = self.observer.quant_min
        self.quant_max = self.observer.quant_max
        self.per_channel = kwargs.get('per_channel', False)
        self.ch_axis = kwargs.get('ch_axis', 0)
        self.enable_observation = True
        self.enable_quantization = True

    def __register_buffer__(self, name, value):
        if hasattr(self, name):
            delattr(self, name)
        self.register_buffer(name, value)

    def __register_parameter__(self, name, value):
        if hasattr(self, name):
            delattr(self, name)
        self.register_parameter(name, nn.Parameter(value))

    def quantize(self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
            round_mode: str = 'nearest'):
        if self.per_channel:
            scale, zero_point = reshape_qparams_by_channel(
                x, scale, zero_point, self.ch_axis)
        if round_mode == 'nearest':
            x_int = RoundSTE.apply(x / scale)
        elif round_mode == 'stochastic':
            x_floor = FloorSTE.apply(x / scale)
            x_int = x_floor + torch.bernoulli((x / scale) - x_floor)
        else: 
            raise NotImplementedError
        x_quant = torch.clamp(x_int + zero_point, self.quant_min, self.quant_max)
        return x_quant
    
    def dequantize(self, x_quant: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
        x_dequant = (x_quant - zero_point) * scale
        return x_dequant
    
    def reset_bitwidth(self, n_bits: int):
        self.observer.reset_bitwidth(n_bits)
        self.quant_min = self.observer.quant_min
        self.quant_max = self.observer.quant_max


class UniformQuantizer(BaseQuantizer):
    def __init__(self, kwargs: dict):
        super().__init__(kwargs)
        self.__register_buffer__('scale', torch.tensor([1.0], dtype=torch.float))
        self.__register_buffer__('zero_point', torch.tensor([0], dtype=torch.int))

    def forward(self, x: torch.Tensor):
        if self.enable_observation:
            x = self.observer(x)
        
        if self.enable_quantization:
            self.scale, self.zero_point = self.observer.calculate_qparams()
            self.scale, self.zero_point = self.scale.to(x.device), self.zero_point.to(x.device)
            x_quant = self.quantize(x, self.scale, self.zero_point)
            x_dequant = self.dequantize(x_quant, self.scale, self.zero_point)
            return x_dequant
        
        return x


class AdaRoundQuantizer(BaseQuantizer):
    def __init__(self, kwargs: dict):
        super().__init__(kwargs)