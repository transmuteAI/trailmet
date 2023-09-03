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
from tqdm import tqdm
from itertools import count
from trailmet.utils import seed_everything
from trailmet.algorithms.quantize.quantize import BaseQuantModel, BaseQuantization
from trailmet.algorithms.quantize.modules import QuantModule, BaseQuantBlock
from trailmet.algorithms.quantize.methods import UniformSymmetricQuantizer, LpNormQuantizer


class QuantModel(BaseQuantModel):
    def __init__(self, model: nn.Module, weight_quant_params: dict, act_quant_params: dict, 
            inplace=False, fuse_model=True):
        super().__init__(model, weight_quant_params, act_quant_params, inplace, fuse_model) 
        self.weight_quantizers = []
        self.act_quantizers = []
        for module in self.model.modules():
            if isinstance(module, QuantModule):
                self.weight_quantizers.append(module.weight_quantizer)
                # if not module.disable_act_quant:
                self.act_quantizers.append(module.act_quantizer)
            elif isinstance(module, BaseQuantBlock):
                self.act_quantizers.append(module.act_quantizer)

    def get_alphas_np(self, weight=True, act=True):
        alphas = []
        quantizers = (self.weight_quantizers if weight else []) + (self.act_quantizers if act else [])
        for quantizer in quantizers:
                alphas.append(quantizer.alpha)
        return torch.tensor(alphas).numpy()
    
    def set_alphas_np(self, alphas: np.ndarray, weight=True, act=True):
        quantizers = (self.weight_quantizers if weight else []) + (self.act_quantizers if act else [])     
        for i, quantizer in enumerate(quantizers):
            quantizer.set_params_from_alpha(alphas[i])
        

class LAPQ(BaseQuantization):
    def __init__(self, model: nn.Module, dataloaders, **kwargs):
        super(LAPQ, self).__init__(**kwargs)
        self.model = model
        self.train_loader = dataloaders['train']
        self.test_loader = dataloaders['test']
        self.kwargs = kwargs
        self.w_bits = kwargs.get('W_BITS', 8)
        self.a_bits = kwargs.get('A_BITS', 8)
        self.calib_batches = kwargs.get('CALIB_BATCHES', 16)
        self.act_quant = kwargs.get('ACT_QUANT', True)
        self.test_before_calibration = kwargs.get('DRY_RUN', True)
        self.maxiter = kwargs.get('MAX_ITER', 1)
        self.maxfev = kwargs.get('MAX_FEV', 1)
        self.verbose = kwargs.get('VERBOSE', True)
        self.print_freq = kwargs.get('PRINT_FREQ', 20)
        self.gpu_id = kwargs.get('GPU_ID', 0)
        self.seed = kwargs.get('SEED', 42)
        seed_everything(self.seed)
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        if self.verbose:
            print("==> Using seed: {} and device: cuda:{}".format(self.seed, self.gpu_id))
        self.calib_data = self.get_calib_samples(self.train_loader, 64*self.calib_batches)
        self.eval_count = count(0)
        self.min_loss = 1e6

    def compress_model(self):
        self.model.to(self.device)
        self.model.eval()

        weight_quant_params = {
            'n_bits': self.w_bits,
            'bcorr': True,
            'method': UniformSymmetricQuantizer,
            'p_val': 2.0,
        }
        act_quant_params = {
            'n_bits': self.a_bits,
            'bcorr': True,
            'method': UniformSymmetricQuantizer,
            'p_val': 2.0,
        }

        if self.test_before_calibration:
            qnn = QuantModel(self.model, weight_quant_params, act_quant_params)
            qnn.set_quant_state(True, True)
            acc1, acc5 = self.test(qnn, self.test_loader, device=self.device)
            print('==> Quantization (W{}A{}) accuracy before LAPQ: {:.4f} | {:.4f}'.format(
                self.w_bits, self.a_bits, acc1, acc5))
            del qnn

        weight_quant_params['method'] = LpNormQuantizer
        act_quant_params['method'] = LpNormQuantizer
        p_vals = np.linspace(2,4,10)
        losses = []
        pbar = tqdm(p_vals, total=len(p_vals))
        for p in pbar:
            weight_quant_params['p_val'] = p
            act_quant_params['p_val'] = p
            qnn = QuantModel(self.model, weight_quant_params, act_quant_params)
            qnn.set_quant_state(True, True)
            loss = self.evaluate_loss(qnn, self.device)
            losses.append(loss.item())
            pbar.set_postfix(p_val=p, loss=loss.item())
            del qnn
        # using quadratic interpolation to approximate the optimal quantization step size ∆p∗
        z = np.polyfit(p_vals, losses, 2)
        y = np.poly1d(z)
        p_intr = y.deriv().roots[0]
        print("==> using p val : {:.2f}  with lp-loss : {:.2f}".format(p_intr, min(losses)))

        weight_quant_params['p_val'] = p_intr
        act_quant_params['p_val'] = p_intr
        self.qnn = QuantModel(self.model, weight_quant_params, act_quant_params)
        self.qnn.set_quant_state(weight_quant=True, act_quant=True)
        lp_acc1, lp_acc5 = self.test(self.qnn, self.test_loader, device=self.device)
        if self.verbose:
            print('==> Quantization (W{}A{}) accuracy before Optimization: {:.4f} | {:.4f}'.format(
                self.w_bits, self.a_bits, lp_acc1, lp_acc5))
            print("==> Starting Powell Optimization")
    
        init_alphas = self.qnn.get_alphas_np()
        
        min_method = "Powell"
        min_options = {
            'maxiter' : self.maxiter,
            'maxfev' : self.maxfev
        }
        count_iter = count(0)
        def local_search_callback(x):
            it = next(count_iter)
            self.qnn.set_alphas_np(x)
            loss = self.evaluate_loss(self.qnn.model, self.device)
            if self.verbose:
                print('\n==> Loss at end of iter [{}] : {:.4f}\n'.format(it, loss.item()))

        self.pbar = tqdm(total=min(self.maxiter, self.maxfev))
        res = optim.minimize(
            lambda alphas: self.evaluate_calibration(alphas, self.qnn, self.device), init_alphas,
            method=min_method, options=min_options, callback=local_search_callback
        )
        self.pbar.close()
        alphas = res.x
        if self.verbose:
            print('==> Layer-wise Scales :\n', alphas)
        self.qnn.set_alphas_np(alphas)
        print('==> Full quantization (W{}A{}) accuracy: {}'.format(
            self.w_bits, self.a_bits, 
            self.test(self.qnn, self.test_loader, device=self.device)))
        return self.qnn


    def evaluate_calibration(self, alphas: np.ndarray, qmodel: QuantModel, device):
        eval_count = next(self.eval_count)
        qmodel.set_alphas_np(alphas)
        loss = self.evaluate_loss(qmodel, device).item()
        if loss < self.min_loss:
            self.min_loss = loss
        self.pbar.set_postfix(curr_loss=loss, min_loss=self.min_loss)
        self.pbar.update(1)
        return loss

    def evaluate_loss(self, model: nn.Module, device):
        criterion = torch.nn.CrossEntropyLoss().to(device)
        model.eval()
        with torch.no_grad():
            if not hasattr(self, 'cal_set'):
                self.cal_set = []
                for i, (images, target) in enumerate(self.train_loader):
                    if i>=self.calib_batches:             # TODO: make this robust for variable batch size
                        break
                    images = images.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    self.cal_set.append((images, target))
            res = torch.tensor([0.]).to(device)
            for i in range(len(self.cal_set)):
                images, target = self.cal_set[i]
                output = model(images)
                loss = criterion(output, target)
                res += loss
            return res / len(self.cal_set)        