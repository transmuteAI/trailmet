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
    def __init__(self, n_bits, symm, channel_wise, delta, zero_point, inited):
        super(BaseQuantizer, self).__init__()
        self._supported_bits = [2, 3, 4, 8, 16, 32]
        assert n_bits in self._supported_bits, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.symmetric = symm
        self.channel_wise = channel_wise
        self.delta = delta
        self.zero_point = zero_point
        self.inited = inited

    def __register_buffer__(self, name, value):
        if hasattr(self, name):
            delattr(self, name)
        self.register_buffer(name, value)

    def __register_parameter__(self, name, value):
        if hasattr(self, name):
            delattr(self, name)
        self.register_parameter(name, nn.Parameter(value))

    def __quantize__(self, x, mode):
        if mode == 'nearest':
            x_int = torch.round(x / self.delta)
        elif mode == 'nearest_ste':
            x_int = RoundSTE.apply(x / self.delta)
        elif mode == 'stochastic':
            x_floor = FloorSTE.apply(x / self.delta)
            x_int = x_floor + torch.bernoulli((x / self.delta) - x_floor)
        else: ValueError('wrong rounding mode')
        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels-1)
        return x_quant
    
    def __dequantize__(self, xq):
        xq_float = (xq - self.zero_point) * self.delta
        return xq_float



class UniformAffineQuantizer(BaseQuantizer):
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
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, **kwargs):
        super(UniformAffineQuantizer, self).__init__(n_bits, symmetric, channel_wise,
            delta=None, zero_point=None, inited=False)
        self.leaf_param = leaf_param
        self.scale_method = scale_method

    def forward(self, x: torch.Tensor):
        if not self.inited:
            delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
            if self.leaf_param:
                self.__register_parameter__('delta', delta)
            else:
                self.__register_buffer__('delta', delta)
            self.__register_buffer__('zero_point', zero_point)
            self.inited = True
        # apply fake quantization
        x_quant = self.__quantize__(x, 'nearest_ste')
        x_dequant = self.__dequantize__(x_quant)
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise = False):
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
                x_max, x_min = x.max(), x.min()
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8
                if self.symmetric:
                    x_absmax = torch.max(x_min.abs(), x_max)
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
                delta = (x_max - x_min) / (self.n_levels - 1)
                zero_point = torch.round(-x_min / delta)

            elif 'mse' in self.scale_method:
                # For Lp norm minimization as described in LAPQ
                x_max, x_min = x.max(), x.min()
                with torch.no_grad():
                    optim_alpha = optim.minimize_scalar(
                        lambda alpha: self.estimate_quant_error(x, x_max, x_min, alpha),
                        bounds=(0.2, 1.0)).x
                delta = optim_alpha * (x_max - x_min) / (self.n_levels - 1)
                zero_point = torch.round( -optim_alpha * x_min / delta)
            else:
                raise NotImplementedError
        return delta, zero_point

    def estimate_quant_error(self, x: torch.Tensor, x_max, x_min, alpha, p=2.4):
        delta = alpha * (x_max - x_min) / (self.n_levels - 1)
        zero_point = torch.round( -alpha * x_min / delta)
        x_int = torch.round(x / delta) # we assume weight quantization is always signed
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_dequant = (x_quant - zero_point) * delta
        err = torch.mean(torch.abs(x_dequant - x) ** p)
        return err.item()

    def bitwidth_refactor(self, refactored_bit: int):
        assert refactored_bit in [2,3,4,8,16,32], 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits
        self.inited = False

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={symmetric}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class AdaRoundQuantizer(BaseQuantizer):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
    Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568
    :param uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor, 
            round_mode='learned_hard_sigmoid', **kwargs):
        # copying all attributes from UniformAffineQuantizer
        super(AdaRoundQuantizer, self).__init__(uaq.n_bits, uaq.symmetric,
            uaq.channel_wise, uaq.delta, uaq.zero_point, inited=True)
        self.round_mode = round_mode
        self.soft_targets = False
        self.__register_buffer__('delta', uaq.delta)
        self.__register_buffer__('zero_point', uaq.zero_point)

        # params for sigmoid function
        self.alpha = None
        self.beta = 2/3
        self.gamma, self.zeta = -0.1, 1.1
        self.init_alpha(x = weight_tensor.clone())

    def forward(self, x: torch.tensor):
        if self.round_mode == 'learned_hard_sigmoid':
            x_floor = FloorSTE.apply(x / self.delta)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).float()
            x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        else:
            x_quant = self.__quantize__(x, mode = self.round_mode)
        x_dequant = self.__dequantize__(x_quant)
        return x_dequant

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x: torch.Tensor):
        x_floor = FloorSTE.apply(x / self.delta)
        if self.round_mode == 'learned_hard_sigmoid':
            rest = (x / self.delta) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.__register_parameter__('alpha', alpha)
        else:
            raise NotImplementedError
        
    def extra_repr(self):
        s = 'bit={n_bits}, round_mode={round_mode}, symmetric={symmetric}, channel_wise={channel_wise}' 
        return s.format(**self.__dict__)


class MaxAbsQuantizer(BaseQuantizer):
    def __init__(self, n_bits, inited=False, **kwargs):
        super().__init__(n_bits, symm=True, channel_wise=False, delta=None, 
            zero_point=None, inited=inited)
        self.alpha = None

    def forward(self, x: torch.Tensor):
        if not self.inited:
            self.init_alpha(x)
        x_quant = self.__quantize__(x, 'nearest_ste')
        x_dequant = self.__dequantize__(x_quant)
        return x_dequant

    def init_alpha(self, x: torch.Tensor):
        alpha = x.abs().max().item()
        self.set_params_from_alpha(alpha)
        self.inited = True

    def set_params_from_alpha(self, alpha):
        self.delta = max((2 * alpha) / (self.n_levels - 1), 1e-8)
        self.zero_point = alpha / self.delta
        self.__register_buffer__('alpha', torch.tensor(alpha))

    def extra_repr(self):
        s = 'bit={n_bits}, alpha={alpha}, inited={inited}' 
        return s.format(**self.__dict__)    

class LpNormQuantizer(MaxAbsQuantizer):
    def __init__(self, n_bits, p_val, inited=False, **kwargs):
        super().__init__(n_bits, inited, **kwargs)
        self.p = p_val
        
    def init_alpha(self, weight_tensor: torch.Tensor):
        with torch.no_grad():
            optim_alpha = optim.minimize_scalar(lambda alpha: self.estimate_quant_error(weight_tensor, alpha),
                bounds=(weight_tensor.min().item(), weight_tensor.max().item())).x
        self.set_params_from_alpha(optim_alpha)
        self.inited = True

    def estimate_quant_error(self, x, alpha):
        delta = max((2 * alpha) / (self.n_levels - 1), 1e-8)
        zero_point = round(alpha / delta)
        x_int = torch.round(x / delta) # we assume weight quantization is always signed
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_dequant = (x_quant - zero_point) * delta
        err = torch.mean(torch.abs(x_dequant - x) ** self.p)
        return err.item()





class BitSplitQuantizer(object):
    def __init__(self, W: np.ndarray, bitwidth):
        self.W = W
        self.bitwidth = bitwidth

    @staticmethod
    def splitWeightInteger(Q, bitwidth):
        """
        Split low-bit weight integers into a list of ternary weights.
        """
        Q_sign = np.sign(Q)
        Q_abs = np.abs(Q)
        B_sav = []
        for idx in range(bitwidth-1):
            B = (Q_abs - Q_abs.astype(np.int)//2*2) # get current last bit
            B *= Q_sign
            B_sav.append(B)
            Q_abs = (Q_abs.astype(np.int)//2).astype(np.float32) # Q_abs >> 1
        return B_sav[::-1]

    @staticmethod
    def splitWeightVector(W, alpha):
        """
        Get the optimal ternary vector, given the quantization scale alpha
        """
        B=W.copy()
        B=(B>=0).astype(np.float32)*2-1
        abs_W2 = np.abs(W) * 2
        B[abs_W2<alpha[:, np.newaxis]] = 0
        return B

    @staticmethod
    def stitch(B_sav):
        """
        Stitch a list of ternary vectors into a integer vectors.
        """
        B_sum = B_sav[0].copy()
        for idx in range(1, len(B_sav)):
            B_sum += B_sav[idx] / (2**idx)
        return B_sum

    @staticmethod
    def stitchExclusive(B_sav, bit):
        """
        Stitch a list of ternary vectors into a integer vectors. The i-th bit is excluded.
        """
        # NOTE: the position of the decimal point is not at the end. E.g. 4-bit fixed-point numbers could be something like:
        # +1.01, -0.11, +1.10, ... instead of +101, -011, +110, ... .
        mask = [1.0]*len(B_sav)
        mask[bit] = 0
        B_sum = B_sav[0] * mask[0]
        for idx in range(1, len(B_sav)):
            B_sum += B_sav[idx] * mask[idx] / (2**idx)
        return B_sum

    def fwa(self):
        """
        Fixed-point Weight Approximation. 
        Minimize the MSE quantization error to find the optimal quantization scales
        Given quantization scale, round-off is used for fixed-point quantization
        """
        max_val = 2**(self.bitwidth-1) - 1
        alpha = np.abs(self.W).max(axis=1) / max_val
        alpha_old = alpha*1.1
        while(np.linalg.norm(alpha-alpha_old)>1e-9):
            q = self.W / alpha[:, np.newaxis]
            q = np.round(q)
            q = np.clip(q, -max_val, max_val)
            alpha_old = alpha
            alpha = np.sum(self.W*q, axis=1) / np.sum(q*q, axis=1)
        return q, alpha

    def ofwa(self, max_epoch=50):
        """
        Optimal Fixed Point Weight Approximation Method.
        Minimize weight matrix reconstruction error using bit-split strategy.
        Given quantization scale, we find the 'optimal' low-bit weights that 
        minimizes the weight quantization error (instead of using round-off).
        Initialized by "fwa".
        """
        assert(2 <= self.bitwidth <= 16)
        Q, alpha = self.fwa()
        B_sav = self.splitWeightInteger(Q, self.bitwidth)
        alpha *= (2**(self.bitwidth-2))  
        # NOTE: the position of the decimal point is not at the end. 
        # E.g. 4-bit fixed-point numbers could be something like:
        # +1.01, -0.11, +1.10, ... instead of +101, -011, +110, ... .
        ### iterative optimization
        for _ in range(max_epoch):
            # print(_)
            alpha_old = np.copy(alpha)
            B_sum = self.stitch(B_sav)
            # given Ws, optimize alpha
            alpha = np.sum(self.W*B_sum, axis=1) / np.sum(B_sum*B_sum, axis=1)
            if np.linalg.norm(alpha_old-alpha) <= 1e-9:
                break
            # given alpha, optimize Ws
            for bit in range(self.bitwidth-1):
                W_res = self.W - self.stitchExclusive(B_sav, bit) * alpha[:, np.newaxis]
                B = self.splitWeightVector(W_res*(2**bit), alpha)
                B_sav[bit] = B
        B_sum = self.stitch(B_sav)
        return B_sav, B_sum, alpha

    def ofwa_rr(self, X: np.ndarray, Y: np.ndarray, max_epoch=100):
        """
        Optimal Fixed Point Weight Approximation with Response Reconstruction.
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
        A = np.dot(X.T, X) # N,N
        for epoch in range(max_epoch):
            # given Bi, optimize alpha
            B_sum = self.stitch(B_sav)
            XB = np.dot(X, B_sum.T) # k,m
            alpha = np.einsum("ij,ij->j", Y, XB)
            alpha = alpha / np.einsum("ij,ij->j", XB, XB)
            # given alpha, optimize Bi
            for bit in range(self.bitwidth-1):
                B = B_sav[bit]
                B_others = self.stitchExclusive(B_sav, bit) * alpha[:, np.newaxis]
                Y_res = Y - np.dot(X, B_others.T)
                T = np.dot(Y_res.T, X) # M,N
                ## fix alpha, optimize B
                # parallel degree: M
                for n in range(N):
                    B[:, n] = 0
                    ABn = np.dot(A[n], B.T)
                    lump = 2 * (ABn * (alpha/(2**bit))- T[:, n]) # M
                    B[:, n] = -np.sign(lump)
                    B[np.abs(lump) < (alpha/(2**bit)) * A[n,n], n] = 0
        B_sum = self.stitch(B_sav)
        return B_sum, alpha

    def ofwa_rr_dw(self, X: np.ndarray, Y: np.ndarray, max_epoch=100):
        '''
        # X: K,M,d,d
        # Y: K,M
        # B: M,N    M kernels
        objective:
        min(Y-XWA)^2
        '''
        # X: M,K,9   (N=d*d)
        B_sav, _, alpha = self.ofwa()
        X = np.transpose(X.reshape(X.shape[0], X.shape[1], -1), (1, 0, 2)) # M, K, 9
        As = np.matmul(np.transpose(X, (0, 2, 1)), X) # M, 9, 9

        alpha_bk = alpha
        for epoch in range(max_epoch):
            # given Bi, optimize alpha
            B_sum = self.stitch(B_sav)
            XB = np.matmul(X, np.expand_dims(B_sum, axis=2)) # M, K, 1
            XB = np.squeeze(XB, axis=2) # M, K
            XB = XB.T

            alpha = np.einsum("ij,ij->j", Y, XB)
            alpha = alpha / np.einsum("ij,ij->j", XB, XB)
            nan_pos = np.isnan(alpha)
            alpha[nan_pos] = alpha_bk[nan_pos]

            # given alpha, optimize Bi
            for bit in range(self.bitwidth-1):
                B = B_sav[bit]
                B_others = self.stitchExclusive(B_sav, bit) * alpha[:, np.newaxis]
                Y_res = Y - np.squeeze(np.matmul(X, np.expand_dims(B_others, axis=2)), axis=2).T # Y_res = Y - np.dot(X, B_others.T)

                T = np.squeeze(np.matmul(np.expand_dims(Y_res.T, axis=1), X), axis=1) #T = np.dot(Y_res.T, X) # M,N
                ## fix alpha, optimize B
                # parallel degree: M
                for n in range(9): # N=9
                    B[:, n] = 0
                    ABn = np.diagonal(np.dot(As[:,n], B.T)) # M #ABn = np.dot(A[n], B.T)
                    lump = 2 * (ABn * (alpha/(2**bit))- T[:, n]) # M
                    B[:, n] = -np.sign(lump)

                    B[np.abs(lump) < (alpha/(2**bit)) * As[:,n,n], n] = 0

        B_sum = self.stitch(B_sav)

        return B_sum, alpha
    


class ActQuantizer(nn.Module):
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
        # return torch.clamp(torch.round(x/self.in_scale), self.min_val, self.max_val) * self.out_scale
        return torch.clamp(RoundSTE.apply(x/self.in_scale), self.min_val, self.max_val) * self.out_scale