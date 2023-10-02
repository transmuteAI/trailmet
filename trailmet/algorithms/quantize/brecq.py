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
from typing import Union
from trailmet.utils import seed_everything
from trailmet.algorithms.quantize.quantize import BaseQuantization, BaseQuantModel
from trailmet.algorithms.quantize.quantize import GetLayerGrad, GetLayerInpOut, BaseQuantLoss
from trailmet.algorithms.quantize.modules import StraightThrough, QuantModule, BaseQuantBlock
from trailmet.algorithms.quantize.methods import UniformAffineQuantizer, AdaRoundQuantizer


class QuantModel(BaseQuantModel):
    def __init__(self, model: nn.Module, weight_quant_params: dict, act_quant_params: dict):
        super(QuantModel, self).__init__(model, weight_quant_params, act_quant_params, fold_bn=True)

    def reset_scale_method(self, scale_method = 'mse', act_quant_reset = False):
        for module in self.quant_modules:
            module.weight_quantizer.scale_method = scale_method
            module.weight_quantizer.inited = False
            if act_quant_reset:
                module.act_quantizer.scale_method = scale_method
                module.act_quantizer.inited = False
    
    def set_head_stem_precision(self, bitwidth):
        """
        Set the precision (bitwidth) for weights and activations for the first and last 
        layers of the model. Also ignore reconstruction for the first layer.
        """
        assert len(self.quant_modules) >= 2, 'Model has less than 2 quantization modules'
        self.quant_modules[0].weight_quantizer.bitwidth_refactor(bitwidth)
        self.quant_modules[0].act_quantizer.bitwidth_refactor(bitwidth)
        self.quant_modules[-1].weight_quantizer.bitwidth_refactor(bitwidth)
        self.quant_modules[-2].act_quantizer.bitwidth_refactor(bitwidth)
        self.quant_modules[0].ignore_reconstruction = True

    def disable_network_output_quantization(self):
        """
        Disable Network Output Quantization
        """
        self.quant_modules[-1].disable_act_quant = True

    

class BRECQ(BaseQuantization):
    """
    Class for post-training quantization using block reconstruction method 
    based on - BRECQ: PUSHING THE LIMIT OF POST-TRAINING QUANTIZATION 
    BY BLOCK RECONSTRUCTION [https://arxiv.org/abs/2102.05426]

    :param W_BITS: bitwidth for weight quantization
    :param A_BITS: bitwidth for activation quantization
    :param CHANNEL_WISE: apply channel_wise quantization for weights
    :param ACT_QUANT: apply activation quantization
    :param SET_8BIT_HEAD_STEM: Set the first and the last layer to 8-bit
    :param NUM_SAMPLES: size of calibration dataset
    :param WEIGHT: weight of rounding cost vs the reconstruction loss
    :param ITERS_W: number of iteration for AdaRound
    :param ITERS_A: number of iteration for LSQ
    :param LR: learning rate for LSQ
    """
    def __init__(self, model: nn.Module, dataloaders, **kwargs):
        super(BRECQ, self).__init__(**kwargs)
        self.model = model
        self.train_loader = dataloaders['train']
        self.test_loader = dataloaders['test']
        self.kwargs = kwargs
        self.w_bits = self.kwargs.get('W_BITS', 8)
        self.a_bits = self.kwargs.get('A_BITS', 8)
        self.channel_wise = self.kwargs.get('CHANNEL_WISE', True)
        self.act_quant = self.kwargs.get('ACT_QUANT', True)
        self.set_8bit_head_stem = self.kwargs.get('SET_8BIT_HEAD_STEM', False)
        self.w_budget = self.kwargs.get('W_BUDGET', None)
        self.use_bits = self.kwargs.get('USE_BITS', [2,4,8])
        self.arch = self.kwargs.get('ARCH', '')
        self.save_path = self.kwargs.get('SAVE_PATH', './runs/')
        self.num_samples = self.kwargs.get('NUM_SAMPLES', 1024)
        self.scale_method = self.kwargs.get('SCALE_METHOD', 'mse')

        self.iters_w = self.kwargs.get('ITERS_W', 10000)
        self.iters_a = self.kwargs.get('ITERS_A', 10000)
        self.optim = self.kwargs.get('OPTIMIZER', torch.optim.adam)
        self.weight = self.kwargs.get('WEIGHT', 0.01)
        self.lr = self.kwargs.get('LR', 4e-5)
        self.p = self.kwargs.get('P_VAL', 2.4)    # Lp norm minimization for LSQ

        self.gpu_id = self.kwargs.get('GPU_ID', 0)
        self.batch_size = self.kwargs.get('BATCH_SIZE', 64)
        self.seed = self.kwargs.get('SEED', 42)
        self.b_start = 20    # temperature at the beginning of calibration
        self.b_end = 2       # temperature at the end of calibration
        self.test_before_calibration = True
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        torch.cuda.set_device(self.gpu_id)
        self.calib_data = self.get_calib_samples(self.train_loader, self.num_samples)
        seed_everything(self.seed)
        print('==> Using seed :',self.seed)


    def compress_model(self):
        """
        method to build quantization parameters and finetune weights and/or activations
        """
        self.model.to(self.device)
        self.model.eval()
        weight_quant_params = {
            'n_bits': self.w_bits, 
            'channel_wise': self.channel_wise, 
            'method': UniformAffineQuantizer,
            'scale_method': self.scale_method,
        }
        act_quant_params = {
            'n_bits': self.a_bits, 
            'channel_wise': False, 
            'method': UniformAffineQuantizer,
            'scale_method': self.scale_method, 
            'leaf_param': self.act_quant,
        }
        self.qnn = QuantModel(self.model, weight_quant_params, act_quant_params)
        self.qnn.to(self.device)
        self.qnn.eval()

        w_compr = self.w_bits/32 if self.w_budget is None else self.w_budget
        if self.w_budget is not None:
            w_bits, qm_size, max_size = self.sensitivity_analysis(
                self.qnn, self.test_loader, self.use_bits, self.w_budget, 
                self.save_path, '{}_{}_{}'.format(self.arch, w_compr, self.a_bits))
            print('==> Found optimal config for approx model size: {:.2f} MB ' \
                ' (orig {:.2f} MB)'.format(qm_size, max_size/self.w_budget))
            self.qnn.set_layer_precision(w_bits, self.a_bits)
            self.qnn.reset_scale_method(self.scale_method, True)
        
        if self.set_8bit_head_stem:
            print('==> Setting the first and the last layer to 8-bit')
            self.qnn.set_head_stem_precision(8)

        self.qnn.set_quant_state(True, False)
        print('==> Initializing weight quantization parameters')
        _ = self.qnn(self.calib_data[:self.batch_size].to(self.device))
        if self.test_before_calibration:
            print('Quantized accuracy before brecq: {}'.format(self.test(self.qnn, self.test_loader, device=self.device)))
        
        # Start quantized weight calibration
        kwargs = dict(
            iters=self.iters_w, 
            opt_mode='mse', 
            act_quant=False, 
            asym=True,
            b_range=(self.b_start, self.b_end), 
            warmup=0.2, 
        )
        print('==> Starting quantized-weight rounding parameter (alpha) calibration')
        self.reconstruct_model(self.qnn, **kwargs)
        self.qnn.set_quant_state(weight_quant=True, act_quant=False)
        print('Weight quantization accuracy: {}'.format(self.test(self.qnn, self.test_loader, device=self.device)))

        if self.act_quant:
            # Initialize activation quantization parameters
            self.qnn.set_quant_state(True, True)
            with torch.no_grad():
                _ = self.qnn(self.calib_data[:self.calib_bs].to(self.device))
            self.qnn.disable_network_output_quantization()
            
            # Start activation rounding calibration
            kwargs = dict(
                iters=self.iters_a, 
                opt_mode='mse', 
                act_quant=True,  
            )
            print('==> Starting quantized-activation scaling parameter (delta) calibration')
            self.reconstruct_model(self.qnn, **kwargs)
            self.qnn.set_quant_state(weight_quant=True, act_quant=True)

            # torch.save(self.qnn.state_dict(), f'{self.save_path}/weights/{self.arch}_{w_compr}_{self.a_bits}.pth')
            print('Full quantization (W{}A{}) accuracy: {}'.format(w_compr, self.a_bits, 
                self.test(self.qnn, self.test_loader, device=self.device))) 
        return self.qnn


    def reconstruct_model(self, module: nn.Module, **kwargs):
        """
        Method for model parameters reconstruction. Takes in quantized model
        and optimizes weights by applying layer-wise reconstruction for first 
        and last layer, and block reconstruction otherwise.
        """
        for name, child_module in module.named_children():
            if isinstance(child_module, QuantModule):
                if child_module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    self.reconstruct_module(self.qnn, child_module, **kwargs)
            elif isinstance(child_module, BaseQuantBlock):
                if child_module.ignore_reconstruction is True:
                    print('Ignore reconstruction of {} block {}'.format(self._parent_name, name))
                    continue
                else:
                    print('Reconstruction for {} block {}'.format(self._parent_name, name))
                    self.reconstruct_module(self.qnn, child_module, **kwargs)
            else:
                self._parent_name = name
                self.reconstruct_model(child_module, **kwargs)


    def reconstruct_module(self, 
            model: BaseQuantModel, module: Union[QuantModule, BaseQuantBlock],
            iters: int = 10000, opt_mode: str = 'mse', act_quant: bool = False, 
            asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2), 
            warmup: float = 0.0):
        
        model.set_quant_state(False, False)
        module.set_quant_state(True, act_quant)
        round_mode = 'learned_hard_sigmoid'
        opt_params = []

        if not include_act_func:
            org_act_func = module.activation_function
            module.activation_function = StraightThrough()

        if not act_quant:
            # Replace weight quantizer to AdaRoundQuantizer and learn alpha
            if isinstance(module, QuantModule):
                module.weight_quantizer = AdaRoundQuantizer(
                    uaq = module.weight_quantizer, round_mode = round_mode,
                    weight_tensor = module.org_weight.data)
                module.weight_quantizer.soft_targets = True
                opt_params.append(module.weight_quantizer.alpha)

            if isinstance(module, BaseQuantBlock):
                for name, submodule in module.named_modules():
                    if isinstance(submodule, QuantModule):
                        submodule.weight_quantizer = AdaRoundQuantizer(
                            uaq = submodule.weight_quantizer, round_mode = round_mode,
                            weight_tensor = submodule.org_weight.data)
                        submodule.weight_quantizer.soft_targets = True
                        opt_params.append(submodule.weight_quantizer.alpha)

            optimizer = self.optim(opt_params)
            scheduler = None
        else:
            # Use UniformAffineQuantizer to learn delta for activations
            if hasattr(module.act_quantizer, 'delta'):
                opt_params.append(module.act_quantizer.delta)
            if isinstance(module, BaseQuantBlock):
                for name, submodule in module.named_modules():
                    if isinstance(submodule, QuantModule) and submodule.act_quantizer.delta is not None:
                        opt_params.append(submodule.act_quantizer.delta)

            optimizer = self.optim(opt_params, lr = self.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)


        loss_mode = 'none' if act_quant else 'relaxation'
        # rec_loss = opt_mode
        loss_func = BaseQuantLoss(
            module, round_loss=loss_mode, weight=self.weight, max_count=iters, 
            rec_loss=opt_mode, b_range=b_range, decay_start=0, warmup=warmup, p=self.p)

        # Save data before optimizing the rounding
        cached_inps, cached_outs = self.save_inp_oup_data(model, module, asym, act_quant)
        if opt_mode != 'mse':
            cached_grads = self.save_grad_data(model, module, act_quant)
        else:
            cached_grads = None

        for i in range(iters):
            idx = torch.randperm(cached_inps.size(0))[:self.batch_size]
            cur_inp = cached_inps[idx].to(self.device)
            cur_out = cached_outs[idx].to(self.device)
            cur_grad = cached_grads[idx].to(self.device) if opt_mode != 'mse' else None

            optimizer.zero_grad()
            out_quant = module(cur_inp)

            err = loss_func(out_quant, cur_out, cur_grad)
            err.backward(retain_graph=True)

            optimizer.step()
            if scheduler:
                scheduler.step()

        torch.cuda.empty_cache()

        # Finish optimization, use hard rounding.
        if isinstance(module, QuantModule):
            module.weight_quantizer.soft_targets = False
        if isinstance(module, BaseQuantBlock):
            for name, submodule in module.named_modules():
                if isinstance(submodule, QuantModule):
                    submodule.weight_quantizer.soft_targets = False

        # Reset original activation function
        if not include_act_func:
            module.activation_function = org_act_func


    def save_inp_oup_data(self, model, layer: Union[QuantModule, BaseQuantBlock],
            asym: bool = False, act_quant: bool = False):
        """
        Function to save input data and output data of a particular layer/block over calibration dataset.
        """
        get_inp_out = GetLayerInpOut(model, layer, device=self.device, asym=asym, act_quant=act_quant)
        cached_batches = []
        torch.cuda.empty_cache()

        for i in range(int(self.calib_data.size(0) / self.batch_size)):
            cur_inp, cur_out = get_inp_out(self.calib_data[i * self.batch_size:(i + 1) * self.batch_size])
            cached_batches.append((cur_inp.cpu(), cur_out.cpu()))

        cached_inps = torch.cat([x[0] for x in cached_batches])
        cached_outs = torch.cat([x[1] for x in cached_batches])
        torch.cuda.empty_cache()

        cached_inps = cached_inps.to(self.device)
        cached_outs = cached_outs.to(self.device)
        return cached_inps, cached_outs
        

    def save_grad_data(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], 
            act_quant: bool = False):
        """
        Function to save gradient data of a particular layer/block over calibration dataset.
        """
        get_grad = GetLayerGrad(model, layer, self.device, act_quant=act_quant)
        cached_batches = []
        torch.cuda.empty_cache()

        for i in range(int(self.calib_data.size(0) / self.batch_size)):
            cur_grad = get_grad(self.calib_data[i * self.batch_size:(i + 1) * self.batch_size])
            cached_batches.append(cur_grad.cpu())

        cached_grads = torch.cat([x for x in cached_batches])
        cached_grads = cached_grads.abs() + 1.0
        # scaling to make sure its mean is 1
        # cached_grads = cached_grads * torch.sqrt(cached_grads.numel() / cached_grads.pow(2).sum())
        torch.cuda.empty_cache()

        cached_grads = cached_grads.to(self.device)
        return cached_grads