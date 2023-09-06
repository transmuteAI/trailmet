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
    def __init__(self, network: str, dataloaders: dict, **kwargs):
        super(LAPQ, self).__init__(**kwargs)
        if network not in supported:
            raise ValueError(f"Network architecture '{network}' not in supported: {supported}")
        self.train_loader = dataloaders['train']
        self.test_loader = dataloaders['test']
        self.kwargs = kwargs
        self.w_bits = kwargs.get('w_bits', 8)
        self.a_bits = kwargs.get('a_bits', 8)
        self.reduce_range = kwargs.get('reduce_range', True)
        self.fake_quantize = kwargs.get('fake_quantize', False)
        self.calib_batches = kwargs.get('calib_batches', 16)
        self.act_quant = kwargs.get('act_quant', True)
        self.test_min_max_quant = kwargs.get('test_min_max_quant', True)
        self.p_val = kwargs.get('p_val', None)
        self.max_iter = kwargs.get('max_iter', 2000)
        self.max_fev = kwargs.get('max_fev', 2000)
        self.eval_freq = kwargs.get('eval_freq', 500)
        self.verbose = kwargs.get('verbose', True)
        self.gpu_id = kwargs.get('gpu_id', 0)
        self.seed = kwargs.get('seed', 42)
        seed_everything(self.seed)
        assert torch.cuda.is_available(), "GPU is required for calibration"
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        if self.verbose:
            print("==> Using seed:{} and device cuda:{}".format(self.seed, self.gpu_id))
        

    def compress_model(self, model: nn.Module, inplace: bool = False):
        model.to(self.device)
        model.eval()

        weight_quant_params = {
            'n_bits': self.w_bits,
            'reduce_range': self.reduce_range,
            'unsigned': False,
            'p_val': 2.0,
            'method': UniformSymmetricQuantizer,
        }
        act_quant_params = {
            'n_bits': self.a_bits,
            'reduce_range': self.reduce_range,
            'unsigned': True,
            'p_val': 2.0,
            'method': UniformSymmetricQuantizer,
        }

        if self.test_min_max_quant:
            acc1, acc5 = self.test(model, self.test_loader, device=self.device, progress=True)
            if self.verbose:
                print('==> Full Precision Model: acc@1 {:.3f} | acc@5 {:.3f}'.format(acc1, acc5))
            qmodel = QuantModel(model, weight_quant_params, act_quant_params)
            qmodel.set_quant_state(True, True)
            acc1, acc5 = self.test(qmodel, self.test_loader, device=self.device, progress=True)
            if self.verbose:
                print('==> Quantization accuracy before LAPQ: acc@1 {:.3f} | acc@5 {:.3f}'.format(acc1, acc5))
            del qmodel

        weight_quant_params['method'] = LpNormQuantizer
        act_quant_params['method'] = LpNormQuantizer

        if self.p_val is None:
            p_vals = np.linspace(2,3.9,20)
            losses = []
            pbar = tqdm(p_vals, total=len(p_vals))
            for p in pbar:
                weight_quant_params['p_val'] = p
                act_quant_params['p_val'] = p
                qmodel = QuantModel(model, weight_quant_params, act_quant_params)
                qmodel.set_quant_state(True, True)
                loss = self.evaluate_loss(qmodel, self.device)
                losses.append(loss.item())
                pbar.set_postfix(p_val=p, loss=loss.item())
                del qmodel
            # using quadratic interpolation to approximate the optimal quantization step size ∆p∗
            z = np.polyfit(p_vals, losses, 2)
            y = np.poly1d(z)
            p_intr = y.deriv().roots[0]
            min_loss = min(losses)
        else:
            weight_quant_params['p_val'] = self.p_val
            act_quant_params['p_val'] = self.p_val
            qmodel = QuantModel(model, weight_quant_params, act_quant_params)
            qmodel.set_quant_state(True, True)
            p_intr = self.p_val
            min_loss = self.evaluate_loss(qmodel, self.device)
            del qmodel

        if self.verbose:
            print("==> using p-val : {:.3f}  with lp-loss : {:.3f}".format(p_intr, min_loss))
        weight_quant_params['p_val'] = p_intr
        act_quant_params['p_val'] = p_intr
        qmodel = QuantModel(model, weight_quant_params, act_quant_params, inplace=inplace)
        qmodel.set_quant_state(weight_quant=True, act_quant=True)
        acc1, acc5 = self.test(qmodel, self.test_loader, device=self.device, progress=True)
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
                self.eval_acc, _ = self.test(qmodel, self.test_loader, device=self.device, progress=False)
                self.eval_iter = it
                # print('\n==> Quantization accuracy at iter [{}]: acc@1 {:.2f} | acc@5 {:.2f}\n'.format(it, acc1, acc5))

        self.min_loss = 1e6
        self.pbar = tqdm(total=min(self.max_iter, self.max_fev))
        res = optim.minimize(
            lambda alphas: self.evaluate_calibration(alphas, qmodel, self.device), init_alphas,
            method=min_method, options=min_options, callback=local_search_callback
        )
        self.pbar.close()
        alphas = res.x
        status = res.success
        if self.verbose:
            print('==> Optimization completed with success status:', status)
            print('==> Optimized alphas :\n', alphas)
        qmodel.set_alphas_np(alphas)
        
        acc1, acc5 = self.test(qmodel, self.test_loader, device=self.device, progress=True)
        if self.verbose:
            print('==> Final LAPQ quantization accuracy: {:.3f} | {:.3f}'.format(acc1, acc5))

        if self.fake_quantize:
            quantized_model = qmodel
        else:
            quantized_model = qmodel.convert_model_to_quantized(inplace=inplace)
        
        return quantized_model


    def evaluate_calibration(self, alphas: np.ndarray, qmodel: QuantModel, device):
        qmodel.set_alphas_np(alphas)
        loss = self.evaluate_loss(qmodel, device).item()
        if loss < self.min_loss:
            self.min_loss = loss
        self.pbar.set_postfix(curr_loss=loss, min_loss=self.min_loss, eval_acc=self.eval_acc, eval_iter=self.eval_iter)
        self.pbar.update(1)
        return loss


    def evaluate_loss(self, model: nn.Module, device):
        criterion = torch.nn.CrossEntropyLoss().to(device)
        model.eval()
        with torch.no_grad():
            if not hasattr(self, 'cal_set'):
                self.cal_set = []
                for i, (images, target) in enumerate(self.train_loader):
                    if i>=self.calib_batches:             
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