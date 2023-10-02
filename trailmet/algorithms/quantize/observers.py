import torch
import torch.nn as nn
import scipy.optimize as optim
from trailmet.algorithms.quantize.utils import get_dtype, get_qscheme, \
    transform_and_flatten_tensor_by_channel, reshape_qparams_by_channel, \
    fake_quantize 

class BaseObserver(nn.Module):
    def __init__(self, n_bits: int = 8, reduce_range: bool = True, unsigned: bool = False, 
            symmetric: bool = False, per_channel: bool = False):
        super(BaseObserver, self).__init__()
        self.n_bits = n_bits
        assert 2 <= n_bits <= 32, "n_bits is outside allowed range [2, 32]"
        
        if reduce_range:
            n_bits -= 1
        if unsigned:
            self.quant_min = 0
            self.quant_max = (2 ** n_bits) - 1
        else:
            self.quant_min = -(2 ** (n_bits - 1))
            self.quant_max = (2 ** (n_bits - 1)) - 1
        
        self.reduce_range = reduce_range
        self.unsigned = unsigned
        self.symmetric = symmetric
        self.per_channel = per_channel
        
        self.eps = torch.tensor(1e-8, dtype=torch.float32)
        self.dtype = get_dtype(self.quant_min, self.quant_max, self.reduce_range)
        self.qscheme = get_qscheme(self.per_channel, self.symmetric)
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        if (unsigned and symmetric and per_channel and reduce_range):
            raise NotImplementedError(
                "cannot reduce range for per-channel-symmetric unsigned quantization"
            )
        self.inited = False

    def reset_bitwidth(self, n_bits):
        self.n_bits = n_bits
        assert 2 <= n_bits <= 32, "n_bits is outside allowed range [2, 32]"
        if self.reduce_range:
            n_bits -= 1
        if self.unsigned:
            self.quant_min = 0
            self.quant_max = (2 ** n_bits) - 1
        else:
            self.quant_min = -(2 ** (n_bits - 1))
            self.quant_max = (2 ** (n_bits - 1)) - 1

    def reset_min_max_vals(self):
        self.min_val.copy_(torch.tensor(float("inf")))
        self.max_val.copy_(torch.tensor(float("-inf")))
        self.inited = False

    def forward(self, x: torch.Tensor):
        # update min_val and max_val from x and make inited true
        return x

    @torch.jit.export
    def _calculate_qparams(self, min_val: torch.Tensor, max_val: torch.Tensor):
        quant_min, quant_max = self.quant_min, self.quant_max
        
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val.device
        scale = torch.ones(min_val.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val.size(), dtype=torch.int64, device=device)

        if self.symmetric:
            abs_max_val = torch.max(-min_val_neg, max_val_pos)
            scale = (2 * abs_max_val) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            if self.unsigned:
                zero_point = zero_point.new_full(zero_point.size(), (quant_min + quant_max) // 2)
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)

        # for scalar values, cast them to tensors of size 1 to keep the shape consistent
        if len(scale.shape) == 0:
            scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
        if len(zero_point.shape) == 0:
            zero_point = torch.tensor([int(zero_point)], dtype=zero_point.dtype, device=device)
        
        return scale, zero_point

    @torch.jit.export
    def calculate_qparams(self):
        assert self.inited, "need to run observation atleast once"
        return self._calculate_qparams(self.min_val, self.max_val)



class MinMaxObserver(BaseObserver):
    def __init__(self, n_bits: int = 8, reduce_range: bool = True, unsigned: bool = False, 
            symmetric: bool = False, per_channel: bool = False, ch_axis: int = 0, **kwargs):
        super().__init__(n_bits, reduce_range, unsigned, symmetric, per_channel)
        self.ch_axis = ch_axis

    def forward(self, x_orig: torch.Tensor):
        if x_orig.numel() == 0:
            return x_orig
        # dtype must match because updates to buffers are done inplace
        x = x_orig.clone().detach().to(self.min_val.dtype)
        
        if self.per_channel:
            y = transform_and_flatten_tensor_by_channel(x, self.ch_axis)
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)      
        else:
            min_val_cur, max_val_cur = torch.aminmax(x)
        
        if not self.inited:
            self.min_val = min_val_cur
            self.max_val = max_val_cur
            self.inited = True
        else:
            self.min_val = torch.min(self.min_val, min_val_cur)
            self.max_val = torch.max(self.max_val, max_val_cur)
        
        return x_orig  



class LpNormObserver(BaseObserver):
    def __init__(self, n_bits: int = 8, reduce_range: bool = True, unsigned: bool = False, 
            symmetric: bool = False, per_channel: bool = False, ch_axis: int = 0,
            p_val: float = 2.4, num_iters: int = 1000, pos_dist: bool = False, **kwargs):
        super().__init__(n_bits, reduce_range, unsigned, symmetric, per_channel)
        self.pos_dist = pos_dist
        self.ch_axis = ch_axis
        self.num_iters = num_iters
        self.p = p_val

    def lp_loss(self, pred: torch.Tensor, trgt: torch.Tensor,
            p: float = 2.4, per_channel: bool = False):
        err = (pred - trgt).abs().pow(p)
        if per_channel:
            err_ = transform_and_flatten_tensor_by_channel(err, self.ch_axis)
            return err_.mean(1)
        else:
            return err.mean()

    def get_quant_loss_from_range(self, x, min_val, max_val):
        scale, zero_point = self._calculate_qparams(min_val, max_val)
        if self.per_channel:
            scale, zero_point = reshape_qparams_by_channel(x, scale, zero_point, self.ch_axis)
        x_q = fake_quantize(x, scale, zero_point, self.quant_min, self.quant_max)
        loss = self.lp_loss(x_q, x, self.p, self.per_channel)
        return loss

    def get_quant_loss_from_alpha(self, x, alpha, x_min, x_max):
        min_val, max_val = x_min * alpha, x_max * alpha
        scale, zero_point = self._calculate_qparams(min_val, max_val)
        x_q = fake_quantize(x, scale, zero_point, self.quant_min, self.quant_max)
        loss = self.lp_loss(x_q, x, self.p, False)
        return loss.item()

    def perform_linear_1D_search(self, x: torch.Tensor):
        pass

    def perform_fast_1D_search(self, x: torch.Tensor):
        if self.per_channel:
            alphas = []
            x_ = transform_and_flatten_tensor_by_channel(x, self.ch_axis)
            x_min, x_max = torch.aminmax(x_, dim=1)
            if self.pos_dist:
                x_min = torch.zeros_like(x_min)

            for ch in range(len(x_)):
                x_ch = x_[ch]
                ch_min, ch_max = x_min[ch], x_max[ch]
                optim_alpha = optim.minimize_scalar(
                    lambda alpha: self.get_quant_loss_from_alpha(x_ch, alpha, ch_min, ch_max),
                    bounds=(0.2, 1.0)).x
                alphas.append(optim_alpha)

            alphas = torch.tensor(alphas, dtype=torch.float32, device=x.device) 
            min_val, max_val = x_min * alphas, x_max * alphas               
        
        else:
            x_min, x_max = torch.aminmax(x)
            if self.pos_dist:
                x_min = torch.zeros_like(x_min)
            optim_alpha = optim.minimize_scalar(
                lambda alpha: self.get_quant_loss_from_alpha(x, alpha, x_min, x_max),
                bounds=(0.2, 1.0)).x 
            min_val, max_val = x_min * optim_alpha, x_max * optim_alpha  

        return min_val, max_val             


    def forward(self, x_orig: torch.Tensor):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        
        if self.symmetric or self.pos_dist:
            min_val_cur, max_val_cur = self.perform_fast_1D_search(x)
        else:
            raise NotImplementedError
        
        if not self.inited:
            self.min_val = min_val_cur
            self.max_val = max_val_cur
            self.inited = True
        else:
            self.min_val = torch.min(self.min_val, min_val_cur)
            self.max_val = torch.max(self.max_val, max_val_cur)
        
        return x_orig 


