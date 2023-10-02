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

supported = [
    "resnet",
    "mobilenetv2"
]

class QuantModel(BaseQuantModel):
    def __init__(self, model: nn.Module, weight_quant_params: dict, act_quant_params: dict, 
            inplace=False, fuse_model=True):
        super().__init__(model, weight_quant_params, act_quant_params, inplace, fuse_model) 
        self.weight_quantizers = []
        self.act_quantizers = []
        self.act_quantizers.append(self.model.inp_quant)
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
    def __init__(self, arch: str, dataloaders: dict, **kwargs):
        super(LAPQ, self).__init__(dataloaders, **kwargs)
        if arch not in supported:
            raise ValueError(f"Network architecture '{arch}' not in supported: {supported}")
        self.kwargs = kwargs
        self.w_bits = kwargs.get('w_bits', 8)
        self.a_bits = kwargs.get('a_bits', 8)
        self.reduce_range = kwargs.get('reduce_range', True)
        self.act_quant = kwargs.get('act_quant', True)
        self.p_val = kwargs.get('p_val', None)
        self.max_iter = kwargs.get('max_iter', 2000)
        self.max_fev = kwargs.get('max_fev', 2000)
        self.eval_freq = kwargs.get('eval_freq', 500)
        self.verbose = kwargs.get('verbose', True)
        calib_bs = kwargs.get('calib_bs', 256)
        calib_size = kwargs.get('calib_size', 1024)
        self.calib_data = self.get_calib_data(calib_bs, calib_size)
        self.seed = kwargs.get('seed', 42)
        seed_everything(self.seed)
        assert torch.cuda.is_available(), "GPU is required for calibration"
        gpu_id = kwargs.get('gpu_id', 0)
        self.device = torch.device('cuda:{}'.format(gpu_id))
        

    def compress_model(self, model: nn.Module, inplace: bool = False, 
            test_before_calibration: bool = True,
            return_fake_quantized: bool = False,
        ) -> nn.Module:
        model.to(self.device)
        model.eval()

        weight_quant_params = {
            'n_bits': self.w_bits,
            'reduce_range': self.reduce_range,
            'unsigned': False,
            'symmetric': True,
            'per_channel': False,
            'quantizer': 'uniform',
            'observer': 'min_max'
        }
        act_quant_params = {
            'n_bits': self.a_bits,
            'reduce_range': self.reduce_range,
            'unsigned': True,
            'symmetric': False,
            'quantizer': 'uniform',
            'observer': 'min_max'
        }

        if test_before_calibration:
            qmodel = QuantModel(model, weight_quant_params, act_quant_params)
            qmodel.set_quantization_state(False, False)
            acc1, acc5 = self.test(qmodel, self.test_data, device=self.device, progress=True)
            if self.verbose:
                print('==> Full Precision Model: acc@1 {:.3f} | acc@5 {:.3f}'.format(acc1, acc5))
            qmodel.set_quantization_state(True, True)
            _ = self.evaluate_loss(qmodel, self.calib_data, self.device)
            qmodel.set_observation_state(False, False)
            acc1, acc5 = self.test(qmodel, self.test_data, device=self.device, progress=True)
            if self.verbose:
                print('==> Quantization accuracy before LAPQ: acc@1 {:.3f} | acc@5 {:.3f}'.format(acc1, acc5))
            del qmodel

        weight_quant_params.update({
            'observer': 'lp_norm',
            'p_val': self.p_val,
        })
        act_quant_params.update({
            'observer': 'lp_norm',
            'p_val': self.p_val,
            'pos_dist': True,
        })

        if self.p_val is None:
            p_vals = np.linspace(2,3.9,20)
            losses = []
            pbar = tqdm(p_vals, total=len(p_vals))
            for p in pbar:
                weight_quant_params['p_val'] = p
                act_quant_params['p_val'] = p
                qmodel = QuantModel(model, weight_quant_params, act_quant_params)
                qmodel.set_quantization_state(True, True)
                loss = self.evaluate_loss(qmodel, self.calib_data, self.device)
                losses.append(loss)
                pbar.set_postfix(p_val=p, loss=loss)
                del qmodel
            # using quadratic interpolation to approximate the optimal ∆p∗
            z = np.polyfit(p_vals, losses, 2)
            y = np.poly1d(z)
            self.p_val = y.deriv().roots[0]

        weight_quant_params['p_val'] = self.p_val
        act_quant_params['p_val'] = self.p_val
        qmodel = QuantModel(model, weight_quant_params, act_quant_params, inplace=inplace)
        qmodel.set_quantization_state(True, True)
        min_loss = self.evaluate_loss(qmodel, self.calib_data, self.device)
        if self.verbose:
            print("==> using p-val : {:.3f}  with lp-loss : {:.3f}".format(self.p_val, min_loss))

        qmodel.set_observation_state(False, False)
        acc1, acc5 = self.test(qmodel, self.test_data, device=self.device, progress=True)
        if self.verbose:
            print('==> Quantization accuracy before optimization: acc@1 {:.3f} | acc@5 {:.3f}'.format(acc1, acc5))
            print("==> Starting Powell Optimization")
    
        init_alphas = qmodel.get_alphas_np()
        min_method = "Powell"
        min_options = {'maxiter' : self.max_iter, 'maxfev' : self.max_fev}

        count_iter = count(0)
        self.eval_acc = 0
        self.eval_iter = 0
        
        def local_search_callback(x):
            it = next(count_iter)
            print(it)
            if self.verbose and it%self.eval_freq==0:
                qmodel.set_alphas_np(x)
                self.eval_acc, _ = self.test(qmodel, self.test_data, device=self.device, progress=False)
                self.eval_iter = it
                # print('\n==> Quantization accuracy at iter [{}]: acc@1 {:.2f} | acc@5 {:.2f}\n'.format(it, acc1, acc5))

        self.min_loss = 1e6
        self.pbar = tqdm(total=min(self.max_iter, self.max_fev))
        res = optim.minimize(
            lambda alphas: self.evaluate_calibration(alphas, qmodel), init_alphas,
            method=min_method, options=min_options, callback=local_search_callback
        )
        self.pbar.close()
        alphas = res.x
        status = res.success
        if self.verbose:
            print('==> Optimization completed with status:', status)
            print('==> Optimized alphas :\n', alphas)
        qmodel.set_alphas_np(alphas)
        
        acc1, acc5 = self.test(qmodel, self.test_data, device=self.device, progress=True)
        if self.verbose:
            print('==> Final LAPQ quantization accuracy: {:.3f} | {:.3f}'.format(acc1, acc5))

        if return_fake_quantized:
            quantized_model = qmodel
        else:
            quantized_model = qmodel.convert_model_to_quantized(inplace=inplace)

        return quantized_model


    def evaluate_calibration(self, alphas: np.ndarray, qmodel: QuantModel):
        qmodel.set_alphas_np(alphas)
        loss = self.evaluate_loss(qmodel, self.calib_data, self.device)
        if loss < self.min_loss:
            self.min_loss = loss
        self.pbar.set_postfix(curr_loss=loss, min_loss=self.min_loss)
        self.pbar.update(1)
        return loss
