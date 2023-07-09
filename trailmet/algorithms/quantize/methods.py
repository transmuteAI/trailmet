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
import numpy as np
import scipy.optimize as optim
import warnings
from trailmet.algorithms.quantize.quantize import BaseQuantization, RoundSTE
from trailmet.utils import lp_loss

__all__ = [
    'UniformAffineQuantizer',
    'AdaRoundQuantizer',
    'BitSplitQuantizer',
    'ActQuantizer',
    'QuantizationBase',
    'UniformQuantization',
    'ClippedUniformQuantization',
    'FixedClipValueQuantization',
    'MaxAbsStaticQuantization',
    'LearnedStepSizeQuantization',
    'LpNormQuantization',
]
"""Quantization classes:-

[BRECQ]
    - UniformAffineQuantizer
    - AdaRoundQuantizer
[BitSplit]
    - BitSplitQuantizer
    - ActQuantizer
[LAPQ]
    - QuantizationBase
        - UniformQuantization
            - ClippedUniformQuantization
                - FixedClipValueQuantization
                - MaxAbsStaticQuantization
                - LearnedStepSizeQuantization
                - LpNormQuantization
"""


class UniformAffineQuantizer(nn.Module):
    """PyTorch Function that can be used for asymmetric quantization (uniform
    affine quantization).

    Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on
    https://arxiv.org/abs/1806.08342.
    Parameters
    ----------
    n_bits: number of bit for quantization
    symmetric: if True, the zero_point should always be 0
    channel_wise: if True, compute scale and zero_point in each channel
    scale_method: determines the quantization scale and zero point
    """

    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        channel_wise: bool = False,
        scale_method: str = 'max',
        leaf_param: bool = False,
    ):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2**self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.leaf_param:
                delta, self.zero_point = self.init_quantization_scale(
                    x, self.channel_wise)
                self.delta = torch.nn.Parameter(delta)
                # self.zero_point = torch.nn.Parameter(self.zero_point)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(
                    x, self.channel_wise)
            self.inited = True

        # start quantization
        # x_int = BQ.round_ste(x / self.delta) + self.zero_point
        x_int = RoundSTE.apply(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant

    def init_quantization_scale(self,
                                x: torch.Tensor,
                                channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(
                    dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(
                    x_clone[c], channel_wise=False)
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
                    warnings.warn(
                        'Quantization range close to zero: [{}, {}]'.format(
                            x_min, x_max))
                    delta = 1e-8

                zero_point = torch.round(-x_min / delta)
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                # For Lp norm minimization as described in LAPQ
                # https://arxiv.org/abs/1911.07190
                x_max = x.max()
                x_min = x.min()
                best_score = 1e10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    score = lp_loss(pred=x, tgt=x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2**self.n_bits - 1)
                        zero_point = torch.round(-new_min / delta)
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2**self.n_bits - 1)
        zero_point = torch.round(-min / delta)
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2**self.n_bits

    def extra_repr(self):
        s = (
            'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},'
            ' leaf_param={leaf_param}')
        return s.format(**self.__dict__)


class AdaRoundQuantizer(nn.Module):
    """Adaptive Rounding Quantizer, used to optimize the rounding policy by
    reconstructing the intermediate output.

    Based on Up or Down? Adaptive Rounding for Post-Training Quantization:
    https://arxiv.org/abs/2004.10568
    https: //arxiv.org/abs/2004.10568
    Parameters
    ----------
    uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    round_mode: controls the forward pass in this quantizer
    weight_tensor: initialize alpha
    """

    def __init__(
        self,
        uaq: UniformAffineQuantizer,
        weight_tensor: torch.Tensor,
        round_mode='learned_round_sigmoid',
    ):
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
        self.beta = 2 / 3
        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        if self.round_mode == 'nearest':
            x_int = torch.round(x / self.delta)
        elif self.round_mode == 'nearest_ste':
            # x_int = BQ.round_ste(x / self.delta)
            x_int = RoundSTE.apply(x / self.delta)
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
        return torch.clamp(
            torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma,
            0, 1)

    def init_alpha(self, x: torch.Tensor):
        x_floor = torch.floor(x / self.delta)
        if self.round_mode == 'learned_hard_sigmoid':
            # print('Init alpha to be FP32')
            rest = (x / self.delta) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log(
                (self.zeta - self.gamma) /
                (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError


class BitSplitQuantizer(object):
    """
    Parameters
    ----------
    W (np.ndarray): Weight vector
    bitwidth (int): bitwidth to be used
    """

    def __init__(self, W: np.ndarray, bitwidth):
        self.W = W
        self.bitwidth = bitwidth

    @staticmethod
    def splitWeightInteger(Q, bitwidth):
        """Split low-bit weight integers into a list of ternary weights."""
        Q_sign = np.sign(Q)
        Q_abs = np.abs(Q)
        B_sav = []
        for idx in range(bitwidth - 1):
            B = Q_abs - Q_abs.astype(np.int) // 2 * 2  # get current last bit
            B *= Q_sign
            B_sav.append(B)
            Q_abs = (Q_abs.astype(np.int) // 2).astype(
                np.float32)  # Q_abs >> 1
        return B_sav[::-1]

    @staticmethod
    def splitWeightVector(W, alpha):
        """Get the optimal ternary vector, given the quantization scale
        alpha."""
        B = W.copy()
        B = (B >= 0).astype(np.float32) * 2 - 1
        abs_W2 = np.abs(W) * 2
        B[abs_W2 < alpha[:, np.newaxis]] = 0
        return B

    @staticmethod
    def stitch(B_sav):
        """Stitch a list of ternary vectors into a integer vectors."""
        B_sum = B_sav[0].copy()
        for idx in range(1, len(B_sav)):
            B_sum += B_sav[idx] / (2**idx)
        return B_sum

    @staticmethod
    def stitchExclusive(B_sav, bit):
        """Stitch a list of ternary vectors into a integer vectors.

        The i-th bit is excluded.
        """
        # NOTE: the position of the decimal point is not at the end. E.g. 4-bit fixed-point numbers could be something like:
        # +1.01, -0.11, +1.10, ... instead of +101, -011, +110, ... .
        mask = [1.0] * len(B_sav)
        mask[bit] = 0
        B_sum = B_sav[0] * mask[0]
        for idx in range(1, len(B_sav)):
            B_sum += B_sav[idx] * mask[idx] / (2**idx)
        return B_sum

    def fwa(self):
        """Fixed-point Weight Approximation.

        Minimize the MSE quantization error to find the optimal quantization
        scales Given quantization scale, round-off is used for fixed-point
        quantization
        """
        max_val = 2**(self.bitwidth - 1) - 1
        alpha = np.abs(self.W).max(axis=1) / max_val
        alpha_old = alpha * 1.1
        while np.linalg.norm(alpha - alpha_old) > 1e-9:
            q = self.W / alpha[:, np.newaxis]
            q = np.round(q)
            q = np.clip(q, -max_val, max_val)
            alpha_old = alpha
            alpha = np.sum(self.W * q, axis=1) / np.sum(q * q, axis=1)
        return q, alpha

    def ofwa(self, max_epoch=50):
        """Optimal Fixed Point Weight Approximation Method.

        Minimize weight matrix reconstruction error using bit-split strategy.
        Given quantization scale, we find the 'optimal' low-bit weights that
        minimizes the weight quantization error (instead of using round-off).
        Initialized by "fwa".
        """
        assert 2 <= self.bitwidth <= 16
        Q, alpha = self.fwa()
        B_sav = self.splitWeightInteger(Q, self.bitwidth)
        alpha *= 2**(self.bitwidth - 2)
        # NOTE: the position of the decimal point is not at the end.
        # E.g. 4-bit fixed-point numbers could be something like:
        # +1.01, -0.11, +1.10, ... instead of +101, -011, +110, ... .
        ### iterative optimization
        for _ in range(max_epoch):
            # print(_)
            alpha_old = np.copy(alpha)
            B_sum = self.stitch(B_sav)
            # given Ws, optimize alpha
            alpha = np.sum(self.W * B_sum, axis=1) / np.sum(B_sum * B_sum,
                                                            axis=1)
            if np.linalg.norm(alpha_old - alpha) <= 1e-9:
                break
            # given alpha, optimize Ws
            for bit in range(self.bitwidth - 1):
                W_res = self.W - self.stitchExclusive(
                    B_sav, bit) * alpha[:, np.newaxis]
                B = self.splitWeightVector(W_res * (2**bit), alpha)
                B_sav[bit] = B
        B_sum = self.stitch(B_sav)
        return B_sav, B_sum, alpha

    def ofwa_rr(self, X: np.ndarray, Y: np.ndarray, max_epoch=100):
        """Optimal Fixed Point Weight Approximation with Response
        Reconstruction.

        Minimize activation matrix reconstruction error using bit-split strategy.
        Initialized by "ofwa".
        :X: K,C,d,d
        :Y: K,M
        :B: M,N    M kernels
        objective:
        min(Y-XWA)^2
        """
        # X: K,N   (N=C*d*d)
        B_sav, _, alpha = self.ofwa()
        X = X.reshape(X.shape[0], -1)
        K, N = X.shape
        A = np.dot(X.T, X)  # N,N
        for epoch in range(max_epoch):
            # given Bi, optimize alpha
            B_sum = self.stitch(B_sav)
            XB = np.dot(X, B_sum.T)  # k,m
            alpha = np.einsum('ij,ij->j', Y, XB)
            alpha = alpha / np.einsum('ij,ij->j', XB, XB)
            # given alpha, optimize Bi
            for bit in range(self.bitwidth - 1):
                B = B_sav[bit]
                B_others = self.stitchExclusive(B_sav, bit) * alpha[:,
                                                                    np.newaxis]
                Y_res = Y - np.dot(X, B_others.T)
                T = np.dot(Y_res.T, X)  # M,N
                ## fix alpha, optimize B
                # parallel degree: M
                for n in range(N):
                    B[:, n] = 0
                    ABn = np.dot(A[n], B.T)
                    lump = 2 * (ABn * (alpha / (2**bit)) - T[:, n])  # M
                    B[:, n] = -np.sign(lump)
                    B[np.abs(lump) < (alpha / (2**bit)) * A[n, n], n] = 0
        B_sum = self.stitch(B_sav)
        return B_sum, alpha

    def ofwa_rr_dw(self, X: np.ndarray, Y: np.ndarray, max_epoch=100):
        """
        # X: K,M,d,d
        # Y: K,M
        # B: M,N    M kernels
        objective:
        min(Y-XWA)^2
        """
        # X: M,K,9   (N=d*d)
        B_sav, _, alpha = self.ofwa()
        X = np.transpose(X.reshape(X.shape[0], X.shape[1], -1),
                         (1, 0, 2))  # M, K, 9
        As = np.matmul(np.transpose(X, (0, 2, 1)), X)  # M, 9, 9

        alpha_bk = alpha
        for epoch in range(max_epoch):
            # given Bi, optimize alpha
            B_sum = self.stitch(B_sav)
            XB = np.matmul(X, np.expand_dims(B_sum, axis=2))  # M, K, 1
            XB = np.squeeze(XB, axis=2)  # M, K
            XB = XB.T

            alpha = np.einsum('ij,ij->j', Y, XB)
            alpha = alpha / np.einsum('ij,ij->j', XB, XB)
            nan_pos = np.isnan(alpha)
            alpha[nan_pos] = alpha_bk[nan_pos]

            # given alpha, optimize Bi
            for bit in range(self.bitwidth - 1):
                B = B_sav[bit]
                B_others = self.stitchExclusive(B_sav, bit) * alpha[:,
                                                                    np.newaxis]
                Y_res = (Y - np.squeeze(
                    np.matmul(X, np.expand_dims(B_others, axis=2)), axis=2).T
                         )  # Y_res = Y - np.dot(X, B_others.T)

                T = np.squeeze(np.matmul(np.expand_dims(Y_res.T, axis=1), X),
                               axis=1)  # T = np.dot(Y_res.T, X) # M,N
                ## fix alpha, optimize B
                # parallel degree: M
                for n in range(9):  # N=9
                    B[:, n] = 0
                    ABn = np.diagonal(np.dot(
                        As[:, n], B.T))  # M #ABn = np.dot(A[n], B.T)
                    lump = 2 * (ABn * (alpha / (2**bit)) - T[:, n])  # M
                    B[:, n] = -np.sign(lump)

                    B[np.abs(lump) < (alpha / (2**bit)) * As[:, n, n], n] = 0

        B_sum = self.stitch(B_sav)

        return B_sum, alpha


class ActQuantizer(nn.Module):
    """
    Parameters
    ----------
    islinear (bool):
    bit_width (int): bit width to be used
    """

    def __init__(self, islinear=False, bit_width=8):
        super(ActQuantizer, self).__init__()
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
            self.min_val = -self.max_val
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
        assert np.min(x) >= 0
        circle_detection_queue = [
            0,
        ] * 5
        alpha = np.max(np.fabs(x)) / self.max_val
        alpha_old = alpha * 0
        n_iter = 0
        circle_detection_queue[n_iter] = alpha
        while np.sum(alpha != alpha_old):
            q = x / alpha
            q = np.clip(np.round(q), self.min_val, self.max_val)
            alpha_old = alpha
            alpha = np.sum(x * q) / np.sum(q * q)
            if alpha in circle_detection_queue:
                break
            n_iter += 1
            circle_detection_queue[n_iter % 5] = alpha
        return alpha

    def forward(self, x):
        if self.in_scale is None:
            assert self.out_scale is None
            return x
        if not isinstance(self.in_scale, (float, np.float32, np.float64)):
            self.in_scale = self.in_scale.to(x.device)
        if not isinstance(self.out_scale, (float, np.float32, np.float64)):
            self.out_scale = self.out_scale.to(x.device)
        # return torch.clamp(torch.round(x/self.in_scale), self.min_val, self.max_val) * self.out_scale
        return (torch.clamp(RoundSTE.apply(x / self.in_scale), self.min_val,
                            self.max_val) * self.out_scale)


class QuantizationBase(object):
    """
    Parameters
    ----------
    module (object): Module to be used
    num_bits (int): Number of bits to be used
    """

    def __init__(self, module, num_bits):
        self.module = module
        self.num_bits = num_bits
        self.num_bins = int(2**num_bits)
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
        learnable_params = [
            d for n, d in params if n in self.learned_parameters()
        ]
        self.opt_params[optim_type + '_' + dataset] = learnable_params

    def optim_parameters(self):
        return self.opt_params

    def loggable_parameters(self):
        return self.named_parameters()

    def named_parameters(self):
        named_params = [(n, p) for n, p in self.named_params
                        if n in self.learned_parameters()]
        return named_params

    @staticmethod
    def learned_parameters():
        return []


class UniformQuantization(QuantizationBase):
    """
    Parameters
    ----------
    module (object): Module to be used
    num_bits (int): Number of bits to be used
    symmetric (bool): Whether the distribution is symmetric or not
    uint (bool):
    stochastic (bool): if True, stochastic rounding will be done
    tails (bool):
    """

    def __init__(self,
                 module,
                 num_bits,
                 symmetric,
                 uint=False,
                 stochastic=False,
                 tails=False):
        super(UniformQuantization, self).__init__(module, num_bits)
        if not symmetric and not uint:
            raise RuntimeError(
                "Can't perform integer quantization on non symmetric distributions."
            )
        self.symmetric = symmetric
        self.uint = uint
        self.stochastic = stochastic
        self.tails = tails
        if uint:
            self.qmax = 2**self.num_bits - 1
            self.qmin = 0
        else:
            self.qmax = 2**(self.num_bits - 1) - 1
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
        return [
            ('bits', self.num_bits),
            ('symmetric', self.symmetric),
            ('tails', self.tails),
        ]

    def __repr__(self):
        s = '{} - ['.format(type(self).__name__)
        for name, value in self.__for_repr__():
            s += '{}: {}, '.format(name, value)
        return s + ']'


class ClippedUniformQuantization(UniformQuantization):
    """
    Parameters
    ----------
    module (object): Module to be used
    num_bits (int): Number of bits to be used
    symmetric (bool): Whether the distribution is symmetric or not
    uint (bool):
    stochastic (bool): if True, stochastic rounding will be done
    tails (bool):
    """

    alpha_param_name = 'alpha'

    def __init__(self,
                 module,
                 num_bits,
                 symmetric,
                 uint=False,
                 stochastic=False,
                 tails=False):
        super(ClippedUniformQuantization,
              self).__init__(module, num_bits, symmetric, uint, stochastic,
                             tails)

    def __call__(self, tensor):
        t_q = self.__quantize__(tensor, self.alpha)
        return t_q

    def __for_repr__(self):
        rpr = super(ClippedUniformQuantization, self).__for_repr__()
        return [(
            self.alpha_param_name,
            '{:.4f}'.format(getattr(self, self.alpha_param_name).item()),
        )] + rpr


class FixedClipValueQuantization(ClippedUniformQuantization):
    """
    Parameters
    ----------
    module (object): Module to be used
    num_bits (int): Number of bits to be used
    symmetric (bool): Whether the distribution is symmetric or not
    uint (bool):
    stochastic (bool): if True, stochastic rounding will be done
    tails (bool):
    kwargs (object): A yaml safe loaded file with information like clip_value, device.
    """

    def __init__(self,
                 module,
                 num_bits,
                 symmetric,
                 uint=False,
                 stochastic=False,
                 kwargs={}):
        super(FixedClipValueQuantization,
              self).__init__(module, num_bits, symmetric, uint, stochastic)
        self.clip_value = kwargs['clip_value']
        self.device = kwargs['device']
        with torch.no_grad():
            self.register_buffer(
                self.alpha_param_name,
                torch.tensor([self.clip_value],
                             dtype=torch.float32).to(self.device),
            )


class MaxAbsStaticQuantization(ClippedUniformQuantization):
    """
    Parameters
    ----------
    module (object): Module to be used
    tensor (torch.Tensor): Tensor which wpuld be quantized
    num_bits (int): Number of bits to be used
    symmetric (bool): Whether the distribution is symmetric or not
    uint (bool):
    stochastic (bool): if True, stochastic rounding will be done
    """

    def __init__(
        self,
        module,
        tensor,
        num_bits,
        symmetric,
        uint=False,
        stochastic=False,
        kwargs={},
    ):
        super(MaxAbsStaticQuantization,
              self).__init__(module, num_bits, symmetric, uint, stochastic)

        with torch.no_grad():
            self.register_buffer(self.alpha_param_name,
                                 tensor.new_tensor([tensor.abs().max()]))


class LearnedStepSizeQuantization(ClippedUniformQuantization):
    """
    Parameters
    ----------
    module (object): Module to be used
    tensor (torch.Tensor): Tensor which wpuld be quantized
    num_bits (int): Number of bits to be used
    symmetric (bool): Whether the distribution is symmetric or not
    uint (bool):
    stochastic (bool): if True, stochastic rounding will be done
    """

    def __init__(self,
                 module,
                 tensor,
                 num_bits,
                 symmetric,
                 uint=False,
                 stochastic=False,
                 **kwargs):
        super(LearnedStepSizeQuantization,
              self).__init__(module, num_bits, symmetric, uint, stochastic)

        with torch.no_grad():
            maxabs = tensor.abs().max()

        self.register_parameter(self.alpha_param_name,
                                tensor.new_tensor([maxabs]))

        self.__create_optim_params__()

    def __create_optim_params__(self):
        # TODO: create default configuration
        self.__add_optim_params__(
            'SGD',
            'imagenet',
            [(
                self.alpha_param_name,
                {
                    'params': [getattr(self, self.alpha_param_name)],
                    'lr': 1e-3,
                    'momentum': 0,
                    'weight_decay': 0,
                },
            )],
        )
        self.__add_optim_params__(
            'SGD',
            'cifar10',
            [(
                self.alpha_param_name,
                {
                    'params': [getattr(self, self.alpha_param_name)],
                    'lr': 1e-1,
                    'momentum': 0,
                    'weight_decay': 0,
                },
            )],
        )

    @staticmethod
    def learned_parameters():
        return [LearnedStepSizeQuantization.alpha_param_name]


class LpNormQuantization(ClippedUniformQuantization):
    """
    Parameters
    ----------
    module (object): Module to be used
    tensor (torch.Tensor): Tensor which wpuld be quantized
    num_bits (int): Number of bits to be used
    symmetric (bool): Whether the distribution is symmetric or not
    uint (bool):
    stochastic (bool): if True, stochastic rounding will be done
    tails (bool):
    kwargs (object): A yaml safe loaded file with information like lp
    """

    def __init__(
        self,
        module,
        tensor,
        num_bits,
        symmetric,
        uint=False,
        stochastic=False,
        tails=False,
        kwargs={},
    ):
        super(LpNormQuantization, self).__init__(module, num_bits, symmetric,
                                                 uint, stochastic, tails)

        self.p = kwargs['lp']
        with torch.no_grad():
            opt_alpha = optim.minimize_scalar(
                lambda alpha: self.estimate_quant_error(alpha, tensor),
                bounds=(tensor.min().item(), tensor.max().item()),
            ).x

        self.register_buffer(self.alpha_param_name,
                             tensor.new_tensor([opt_alpha]))

    def estimate_quant_error(self, alpha, x):
        xq = self.__quantize__(x, alpha)
        err = torch.mean(torch.abs(xq - x)**self.p)
        return err.item()
