import torch
import torch.nn as nn
import scipy.optimize as optim
from trailmet.algorithms.quantize.quantize import RoundSTE


class QuantizationBase(object):
    def __init__(self, module, num_bits):
        self.module = module
        self.num_bits = num_bits
        self.num_bins = int(2 ** num_bits)
        self.opt_params = {}
        self.named_params = []

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

    def __add_optim_params__(self, optim_type, dataset, params):
        learnable_params = [d for n, d in params if n in self.learned_parameters()]
        self.opt_params[optim_type + '_' + dataset] = learnable_params

    def optim_parameters(self):
        return self.opt_params

    def loggable_parameters(self):
        return self.named_parameters()

    def named_parameters(self):
        named_params = [(n, p) for n, p in self.named_params if n in self.learned_parameters()]
        return named_params

    @staticmethod
    def learned_parameters():
        return []

class UniformQuantization(QuantizationBase):
    def __init__(self, module, num_bits, symmetric, uint=False, stochastic=False, tails=False):
        super(UniformQuantization, self).__init__(module, num_bits)
        if not symmetric and not uint:
            raise RuntimeError("Can't perform integer quantization on non symmetric distributions.")
        self.symmetric = symmetric
        self.uint = uint
        self.stochastic = stochastic
        self.tails = tails
        if uint:
            self.qmax = 2 ** self.num_bits - 1
            self.qmin = 0
        else:
            self.qmax = 2 ** (self.num_bits - 1) - 1
            self.qmin = -self.qmax - 1
        if tails:
            self.qmax -= 0.5 + 1e-6
            self.qmin -= 0.5

    def __quantize__(self, tensor, alpha):
        delta = (2 if self.symmetric else 1) * alpha / (self.num_bins - 1)
        delta = max(delta, 1e-8)
        # quantize
        if self.uint and self.symmetric:
            t_q = (tensor + alpha) / delta
        else:
            t_q = tensor / delta
        # stochastic rounding
        if self.stochastic and self.module.training:
            with torch.no_grad():
                noise = t_q.new_empty(t_q.shape).uniform_(-0.5, 0.5)
                t_q += noise
        # clamp and round
        t_q = torch.clamp(t_q, self.qmin, self.qmax)
        t_q = RoundSTE.apply(t_q)
        assert torch.unique(t_q).shape[0] <= self.num_bins
        # de-quantize
        if self.uint and self.symmetric:
            t_q = t_q * delta - alpha
        else:
            t_q = t_q * delta
        return t_q

    def __quantize_gemmlowp__(self, tensor, min_, max_):
        assert self.uint is True
        delta = (max_ - min_) / (self.num_bins - 1)
        delta = max(delta, 1e-8)
        # quantize
        t_q = (tensor - min_) / delta
        # stochastic rounding
        if self.stochastic and self.module.training:
            with torch.no_grad():
                noise = t_q.new_empty(t_q.shape).uniform_(-0.5, 0.5)
                t_q += noise
        # clamp and round
        t_q = torch.clamp(t_q, self.qmin, self.qmax)
        t_q = RoundSTE.apply(t_q)
        assert torch.unique(t_q).shape[0] <= self.num_bins
        # de-quantize
        t_q = t_q * delta + min_
        return t_q

    def __for_repr__(self):
        return [('bits', self.num_bits), ('symmetric', self.symmetric), ('tails', self.tails)]

    def __repr__(self):
        s = '{} - ['.format(type(self).__name__)
        for name, value in self.__for_repr__():
            s += '{}: {}, '.format(name, value)
        return s + ']'

class ClippedUniformQuantization(UniformQuantization):
    alpha_param_name = 'alpha'

    def __init__(self, module, num_bits, symmetric, uint=False, stochastic=False, tails=False):
        super(ClippedUniformQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic,tails)

    def __call__(self, tensor):
        t_q = self.__quantize__(tensor, self.alpha)
        return t_q

    def __for_repr__(self):
        rpr = super(ClippedUniformQuantization, self).__for_repr__()
        return [(self.alpha_param_name, '{:.4f}'.format(getattr(self, self.alpha_param_name).item()))] + rpr


class FixedClipValueQuantization(ClippedUniformQuantization):
    def __init__(self, module, num_bits, symmetric, uint=False, stochastic=False, kwargs={}):
        super(FixedClipValueQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic)
        self.clip_value = kwargs['clip_value']
        self.device = kwargs['device']
        with torch.no_grad():
            self.register_buffer(self.alpha_param_name, torch.tensor([self.clip_value], dtype=torch.float32).to(self.device))


class MaxAbsStaticQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, kwargs={}):
        super(MaxAbsStaticQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic)

        with torch.no_grad():
            self.register_buffer(self.alpha_param_name, tensor.new_tensor([tensor.abs().max()]))
            

class LearnedStepSizeQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, **kwargs):
        super(LearnedStepSizeQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic)

        with torch.no_grad():
            maxabs = tensor.abs().max()

        self.register_parameter(self.alpha_param_name, tensor.new_tensor([maxabs]))

        self.__create_optim_params__()

    def __create_optim_params__(self):
        # TODO: create default configuration
        self.__add_optim_params__('SGD', 'imagenet', [
            (self.alpha_param_name, {'params': [getattr(self, self.alpha_param_name)], 'lr': 1e-3, 'momentum': 0, 'weight_decay': 0})
        ])
        self.__add_optim_params__('SGD', 'cifar10', [
            (self.alpha_param_name, {'params': [getattr(self, self.alpha_param_name)], 'lr': 1e-1, 'momentum': 0, 'weight_decay': 0})
        ])

    @staticmethod
    def learned_parameters():
        return [LearnedStepSizeQuantization.alpha_param_name]


class LpNormQuantization(ClippedUniformQuantization):
    def __init__(self, module, tensor, num_bits, symmetric, uint=False, stochastic=False, tails=False, kwargs={}):
        super(LpNormQuantization, self).__init__(module, num_bits, symmetric, uint, stochastic, tails)

        self.p = kwargs['lp']
        with torch.no_grad():
            opt_alpha = optim.minimize_scalar(lambda alpha: self.estimate_quant_error(alpha, tensor),
                                            bounds=(tensor.min().item(), tensor.max().item())).x

        self.register_buffer(self.alpha_param_name, tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        xq = self.__quantize__(x, alpha)
        err = torch.mean(torch.abs(xq - x) ** self.p)
        return err.item()