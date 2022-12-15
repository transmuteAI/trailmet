import torch
import torch.nn as nn
import scipy.optimize as opt
import warnings
from trailmet.algorithms.quantize.quantize import RoundSTE, BaseQuantization as BQ



class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (uniform affine quantization). 
    Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = True, scale_method: str = 'mse',
                 leaf_param: bool = False, **kwargs):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            if self.leaf_param:
                delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                self.delta = torch.nn.Parameter(delta)
                # self.zero_point = torch.nn.Parameter(self.zero_point)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        # start quantization
        x_int = BQ.round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                delta = float(x_max - x_min) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = torch.round(-x_min / delta)
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                # For Lp norm minimization as described in LAPQ
                # https://arxiv.org/abs/1911.07190
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    score = BQ.lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = torch.round(- new_min / delta)
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = torch.round(- min / delta)
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
    Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568
    :param uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor, round_mode='learned_round_sigmoid'):
        super(AdaRoundQuantizer, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels

        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False

        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        if self.round_mode == 'nearest':
            x_int = torch.round(x / self.delta)
        elif self.round_mode == 'nearest_ste':
            x_int = BQ.round_ste(x / self.delta)
        elif self.round_mode == 'stochastic':
            x_floor = torch.floor(x / self.delta)
            rest = (x / self.delta) - x_floor  # rest of rounding
            x_int = x_floor + torch.bernoulli(rest)
            print('Draw stochastic sample')
        elif self.round_mode == 'learned_hard_sigmoid':
            x_floor = torch.floor(x / self.delta)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise ValueError('Wrong rounding mode')

        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - self.zero_point) * self.delta

        return x_float_q

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x: torch.Tensor):
        x_floor = torch.floor(x / self.delta)
        if self.round_mode == 'learned_hard_sigmoid':
            # print('Init alpha to be FP32')
            rest = (x / self.delta) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError



class UniformQuantization(object):
    param_name = 'alpha'
    def __init__(self, module, **kwargs):
        super(UniformQuantization, self).__init__()
        self.module = module
        self.n_bits = kwargs.get('num_bits', 8)
        assert 2 <= self.n_bits <= 8, 'bitwidth not supported'
        self.n_bins = int(2**self.n_bits)       # n_levels
        self.symm = kwargs.get('symm', True)
        self.uint = kwargs.get('uint', True)
        if not self.symm and not self.uint:
            raise RuntimeError("Cannot perform integer quantization on asymmetric distribution")
        self.stochastic = kwargs.get('stochastic', False)    # To do : add optional stochastic noise in __quantize__
        self.tails = kwargs.get('tails', False)
        if self.uint:
            self.qmax = 2 ** self.n_bits - 1
            self.qmin = 0
        else:
            self.qmax = 2 ** (self.n_bits - 1) - 1
            self.qmin = -self.qmax - 1
        if self.tails:
            self.qmax -= 0.5 + 1e-6
            self.qmin -= 0.5
        self.named_params = []

    def __quantize__(self, tensor, alpha):
        delta = (2 if self.symm else 1) * alpha / (self.n_bins - 1)
        delta = max(delta, 1e-8)
        # quantize
        if self.uint and self.symm:
            q_tensor = (tensor + alpha) / delta
        else:
            q_tensor = tensor / delta
        # clamp and round
        q_tensor = torch.clamp(q_tensor, self.qmin, self.qmax)
        q_tensor = RoundSTE.apply(q_tensor)
        assert torch.unique(q_tensor).shape[0] <= self.n_bins, 'clamping unsuccessful'
        # de-quantize
        if self.uint and self.symm:
            q_tensor = q_tensor * delta - alpha
        else:
            q_tensor = q_tensor * delta
        return q_tensor

    def __repr__(self):
        rpr = [('bits', self.n_bits), ('symmetric', self.symm), ('tails', self.tails)]
        return [(self.param_name, '{:.4f}'.format(getattr(self, self.param_name).item()))] + rpr
    
    def register_buffer(self, name, value):
        if hasattr(self.module, name):
            delattr(self.module, name)
        self.module.register_buffer(name, value)
        setattr(self, name, getattr(self.module, name))

    def register_parameter(self, name, value):
        if hasattr(self.module, name):
            delattr(self.module, name)
        self.module.register_parameter(name, nn.Parameter(value))
        setattr(self, name, getattr(self.module, name))

        self.named_params.append((name, getattr(self.module, name)))


class LpNormQuantizer(UniformQuantization):
    def __init__(self, module, tensor, **kwargs):
        super(LpNormQuantizer, self).__init__(module, **kwargs)
        self.p = kwargs.get('lp', 2.4)
        with torch.no_grad():
            opt_alpha = opt.minimize_scalar(lambda alpha: self.estimate_quant_error(
                alpha, tensor), bounds=(tensor.min().item(), tensor.max().item())).x
        self.register_buffer(self.param_name, tensor.new_tensor([opt_alpha]))

    def __call__(self, tensor):
        t_q = self.__quantize__(tensor, self.alpha)
        return t_q

    def estimate_quant_error(self, alpha, x):
        xq = self.__quantize__(x, alpha)
        err = torch.mean(torch.abs(xq-x)**self.p)
        return err.item()


class FixQuantizationClipValue(UniformQuantization):
    def __init__(self, module, clip_value, device, **kwargs):
        super(FixQuantizationClipValue, self).__init__(module, **kwargs)
        self.clip_value = clip_value
        self.device = device
        assert self.clip_value is not None, 'missing parameter - clip_value'
        assert self.device is not None, 'missing parameter - device'
        with torch.no_grad():
            self.register_buffer(self.param_name, torch.tensor([self.clip_value], dtype=torch.float32).to(self.device))

    def __call__(self, tensor):
        t_q = self.__quantize__(tensor, self.alpha)
        return t_q