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
# source: https://github.com/yhhhli/BRECQ/tree/main/quant

import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Union
from tqdm import tqdm
from trailmet.algorithms.quantize.quantize import (
    StraightThrough,
    BaseQuantization as BQ,
)
from trailmet.algorithms.quantize.qmodel import QuantModule, BaseQuantBlock
from trailmet.algorithms.quantize.methods import AdaRoundQuantizer
from trailmet.utils import lp_loss

__all__ = [
    'StopForwardException',
    'DataSaverHook',
    'GetLayerInpOut',
    'save_inp_oup_data',
    'GradSaverHook',
    'GetLayerGrad',
    'save_grad_data',
    'LinearTempDecay',
    'LayerLossFunction',
    'layer_reconstruction',
    'BlockLossFunction',
    'block_reconstruction',
]

optimizer_map = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adagrad': torch.optim.Adagrad,
    'adadelta': torch.optim.Adadelta,
}


class StopForwardException(Exception):
    """Used to throw and catch an exception to stop traversing the graph."""

    pass


class DataSaverHook:
    """Forward hook that stores the input and output of a layer.

    Parameters
    ----------
    store_input (bool): If True, input of a layer will be saved, default=False
    store_output (bool): If True, output of a layer will be saved, default=False
    stop_forward (bool): If True, forward prop will be stopped, default=False.
    """

    def __init__(self,
                 store_input=False,
                 store_output=False,
                 stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class GetLayerInpOut:
    """Get the input and output of a specified layer in a quantized model.

    model: quantized model for which the input and output needs to be extracted.
    layer: the layer for which input and output needs to be extracted.
    device: the device on which the computation needs to be performed.
    asym: save quantized input and full precision output. [default=False]
    act_quant: use activation quantization. [default=False]
    """

    def __init__(
        self,
        model,
        layer: Union[QuantModule, BaseQuantBlock],
        device: torch.device,
        asym: bool = False,
        act_quant: bool = False,
    ):
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = device
        self.act_quant = act_quant
        self.data_saver = DataSaverHook(store_input=True,
                                        store_output=True,
                                        stop_forward=True)

    def __call__(self, model_input):
        """
        Parameters
        ----------
        model_input: calibration data samples
        return: tuple of layer input and output
        """
        self.model.eval()
        self.model.set_quant_state(False, False)

        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            try:
                _ = self.model(model_input.to(self.device))
            except StopForwardException:
                pass

            if self.asym:
                self.data_saver.store_output = False
                self.model.set_quant_state(weight_quant=True,
                                           act_quant=self.act_quant)
                try:
                    _ = self.model(model_input.to(self.device))
                except StopForwardException:
                    pass
                self.data_saver.store_output = True

        handle.remove()

        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()

        return (
            self.data_saver.input_store[0].detach(),
            self.data_saver.output_store.detach(),
        )


def save_inp_oup_data(
    model,
    layer: Union[QuantModule, BaseQuantBlock],
    cali_data: torch.Tensor,
    asym: bool = False,
    act_quant: bool = False,
    batch_size: int = 32,
    keep_gpu: bool = True,
):
    """Function to save input data and output data of a particular layer/block
    over calibration dataset.

    Parameters
    ----------
    model: quantized model for which the input and output needs to be extracted.
    layer: the layer for which input and output needs to be extracted.
    cali_data: calibration dataset
    asym: save quantized input and full precision output. [default=False]
    act_quant: use activation quantization. [default=False]
    batch_size: mini-batch size for calibration. [default=32]
    keep_gpu: put saved data on GPU for faster optimization. [default=True]
    :return: input and output data
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model,
                                 layer,
                                 device=device,
                                 asym=asym,
                                 act_quant=act_quant)
    cached_batches = []
    torch.cuda.empty_cache()

    for i in range(int(cali_data.size(0) / batch_size)):
        cur_inp, cur_out = get_inp_out(cali_data[i * batch_size:(i + 1) *
                                                 batch_size])
        cached_batches.append((cur_inp.cpu(), cur_out.cpu()))

    cached_inps = torch.cat([x[0] for x in cached_batches])
    cached_outs = torch.cat([x[1] for x in cached_batches])
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)
    return cached_inps, cached_outs


class GradSaverHook:
    """Backward hook that stores the gradients of a layer.

    Parameters
    ----------
    store_grad (bool): if True, gradient of the layer will be stored
    """

    def __init__(self, store_grad=True):
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = None

    def __call__(self, module, grad_input, grad_output):
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException


class GetLayerGrad:
    """Get the gradient a specified layer in a quantized model.

    Parameters
    ----------
    model: quantized model for which the input and output needs to be extracted.
    layer: the layer for which input and output needs to be extracted.
    device: the device on which the computation needs to be performed.
    asym: if True, save quantized input and full precision output. [default=False]
    act_quant: use activation quantization. [default=False]
    """

    def __init__(
        self,
        model,
        layer: Union[QuantModule, BaseQuantBlock],
        device: torch.device,
        act_quant: bool = False,
    ):
        self.model = model
        self.layer = layer
        self.device = device
        self.act_quant = act_quant
        self.data_saver = GradSaverHook(True)

    def __call__(self, model_input):
        """Compute the gradients of layer output, note that we compute the
        gradient by calculating the KL loss between fp model and quant model.

        Parameters
        ----------
        model_input: calibration data samples
        :return: gradients for the layer
        """
        self.model.eval()

        handle = self.layer.register_backward_hook(self.data_saver)
        with torch.enable_grad():
            try:
                self.model.zero_grad()
                inputs = model_input.to(self.device)
                self.model.set_quant_state(False, False)
                out_fp = self.model(inputs)
                self.model.quantize_model_till(self.layer, self.act_quant)
                out_q = self.model(inputs)
                loss = F.kl_div(
                    F.log_softmax(out_q, dim=1),
                    F.softmax(out_fp, dim=1),
                    reduction='batchmean',
                )
                loss.backward()
            except StopForwardException:
                pass

        handle.remove()
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()
        return self.data_saver.grad_out.data


def save_grad_data(
    model,
    layer: Union[QuantModule, BaseQuantBlock],
    cali_data: torch.Tensor,
    damping: float = 1.0,
    act_quant: bool = False,
    batch_size: int = 32,
    keep_gpu: bool = True,
):
    """Function to save gradient data of a particular layer/block over
    calibration dataset.

    Parameters
    ----------
    model: quantized model for which the input and output needs to be extracted.
    layer: the layer for which input and output needs to be extracted.
    cali_data: calibration dataset
    damping: damping the second-order gradient by adding some constant in the FIM diagonal
    act_quant: use activation quantization. [default=False]
    batch_size: mini-batch size for calibration. [default=32]
    keep_gpu: put saved data on GPU for faster optimization. [default=True]
    :return: gradient data
    """
    device = next(model.parameters()).device
    get_grad = GetLayerGrad(model, layer, device, act_quant=act_quant)
    cached_batches = []
    torch.cuda.empty_cache()

    for i in range(int(cali_data.size(0) / batch_size)):
        cur_grad = get_grad(cali_data[i * batch_size:(i + 1) * batch_size])
        cached_batches.append(cur_grad.cpu())

    cached_grads = torch.cat([x for x in cached_batches])
    cached_grads = cached_grads.abs() + 1.0
    # scaling to make sure its mean is 1
    # cached_grads = cached_grads * torch.sqrt(cached_grads.numel() / cached_grads.pow(2).sum())
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_grads = cached_grads.to(device)
    return cached_grads


# ================================
# ****** Reconstruct Layer *******
# ================================


class LinearTempDecay:
    """Class to implement a linear temperature decay scheduler for a given
    maximum time step.

    Parameters
    ----------
    t_max: maximum number of time steps to decay temperature over.
    rel_start_decay: relative point in time to start the decay from the maximum time step. [default=.2]
    start_b: initial temperature value. [default=10]
    end_b: final temperature value. [default=2]
    """

    def __init__(
        self,
        t_max: int,
        rel_start_decay: float = 0.2,
        start_b: int = 10,
        end_b: int = 2,
    ):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """Cosine annealing scheduler for temperature b.

        Parameters
        ----------
        t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(
                0.0, (1 - rel_t))


class LayerLossFunction:
    """
    Parameters
    ----------
    layer (object): layer to be quantized
    Round_loss (str): type of regularization term used to optimize rounding policy (options: relaxation, none)
    Weight (float): weight of rounding loss in total loss
    Rec_loss (str): type of output reconstruction loss (options: mse, fisher_diag, fisher_full)
    max_count (int): number of iterations
    b_range (tuple): range of rounding relaxation factor (b) with linear temp decay scheduler
    decay_start (float): starting point for temp decay of b
    warmup (float): fraction of iterations used for warmup before applying rounding loss
    p (float): power in lp-norm computation of reconstruction loss
    """

    def __init__(
        self,
        layer: QuantModule,
        round_loss: str = 'relaxation',
        weight: float = 1.0,
        rec_loss: str = 'mse',
        max_count: int = 2000,
        b_range: tuple = (10, 2),
        decay_start: float = 0.0,
        warmup: float = 0.0,
        p: float = 2.0,
    ):
        self.layer = layer
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.count = 0
        # self.pbar = tqdm(total=max_count)
        self.pbar = tqdm(
            total=max_count,
            desc='Reconstructing Layer: Loss (X.X) b (X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
        )
        self.temp_decay = LinearTempDecay(
            max_count,
            rel_start_decay=warmup + (1 - warmup) * decay_start,
            start_b=b_range[0],
            end_b=b_range[1],
        )

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        Parameters
        ----------
        pred: output from quantized model
        tgt: output from FP model
        grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError(
                'Not supported reconstruction loss function: {}'.format(
                    self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            round_vals = self.layer.weight_quantizer.get_soft_targets()
            round_loss += (self.weight *
                           (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum())
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss

        if self.count % 100 == 0:
            self.pbar.set_description(
                'Reconstructing Layer: Loss ({:.3f}) b ({:.1f})'.format(
                    float(total_loss), b))
        #     self.pbar.set_postfix(loss=float(total_loss), b=b)
        self.pbar.update(1)
        return total_loss


def layer_reconstruction(
    model,
    layer: QuantModule,
    cali_data: torch.Tensor,
    batch_size: int = 32,
    iters: int = 20000,
    weight: float = 0.001,
    opt_mode: str = 'mse',
    asym: bool = False,
    include_act_func: bool = True,
    b_range: tuple = (20, 2),
    warmup: float = 0.0,
    act_quant: bool = False,
    lr: float = 4e-5,
    p: float = 2.0,
    multi_gpu: bool = False,
    optim='adam',
):
    """Block reconstruction to optimize the output from each layer.

    Parameters
    ----------
    model: QuantModel
    layer: QuantModule that needs to be optimized
    cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    batch_size: mini-batch size for reconstruction
    iters: optimization iterations for reconstruction,
    weight: the weight of rounding regularization term
    opt_mode: optimization mode
    asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    include_act_func: optimize the output after activation function
    b_range: temperature range
    warmup: proportion of iterations that no scheduling for temperature
    act_quant: use activation quantization or not.
    lr: learning rate for act delta learning
    p: L_p norm minimization
    multi_gpu: use multi-GPU or not, if enabled, we should sync the gradients
    """

    model.set_quant_state(False, False)
    layer.set_quant_state(True, act_quant)
    round_mode = 'learned_hard_sigmoid'

    if not include_act_func:
        org_act_func = layer.activation_function
        layer.activation_function = StraightThrough()

    if not act_quant:
        # Replace weight quantizer to AdaRoundQuantizer
        layer.weight_quantizer = AdaRoundQuantizer(
            uaq=layer.weight_quantizer,
            round_mode=round_mode,
            weight_tensor=layer.org_weight.data,
        )
        layer.weight_quantizer.soft_targets = True

        # Set up optimizer
        opt_params = [layer.weight_quantizer.alpha]
        optimizer = optimizer_map[optim](opt_params)
        scheduler = None
    else:
        # Use UniformAffineQuantizer to learn delta
        opt_params = [layer.act_quantizer.delta]
        optimizer = optimizer_map[optim](opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=iters,
                                                               eta_min=0.0)

    loss_mode = 'none' if act_quant else 'relaxation'
    rec_loss = opt_mode

    loss_func = LayerLossFunction(
        layer,
        round_loss=loss_mode,
        weight=weight,
        max_count=iters,
        rec_loss=rec_loss,
        b_range=b_range,
        decay_start=0,
        warmup=warmup,
        p=p,
    )

    # Save data before optimizing the rounding
    cached_inps, cached_outs = save_inp_oup_data(model, layer, cali_data, asym,
                                                 act_quant, batch_size)
    if opt_mode != 'mse':
        cached_grads = save_grad_data(model,
                                      layer,
                                      cali_data,
                                      act_quant,
                                      batch_size=batch_size)
    else:
        cached_grads = None
    device = 'cuda'
    for i in range(iters):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]
        cur_inp = cached_inps[idx]
        cur_out = cached_outs[idx]
        cur_grad = cached_grads[idx] if opt_mode != 'mse' else None

        optimizer.zero_grad()
        out_quant = layer(cur_inp)

        err = loss_func(out_quant, cur_out, cur_grad)
        err.backward(retain_graph=True)
        if multi_gpu:
            for p in opt_params:
                dist.all_reduce(p.grad)
        optimizer.step()
        if scheduler:
            scheduler.step()

    torch.cuda.empty_cache()

    # Finish optimization, use hard rounding.
    layer.weight_quantizer.soft_targets = False

    # Reset original activation function
    if not include_act_func:
        layer.activation_function = org_act_func


# =================================
# ******* Reconstruct Block *******
# =================================


class BlockLossFunction:
    """
    Parameters
    ----------
    Module (object): module or block being quantized
    Round_loss (str): type of regularization term used to optimize rounding policy (options: relaxation, none)
    Weight (float): weight of rounding loss in total loss
    Rec_loss (str): type of output reconstruction loss (options: mse, fisher_diag, fisher_full)
    max_count (int): number of iterations
    b_range (tuple): range of rounding relaxation factor (b) with linear temp decay scheduler
    decay_start (float): starting point for temp decay of b
    warmup (float): fraction of iterations used for warmup before applying rounding loss
    p (float): power in lp-norm computation of reconstruction loss
    """

    def __init__(
        self,
        block: BaseQuantBlock,
        round_loss: str = 'relaxation',
        weight: float = 1.0,
        rec_loss: str = 'mse',
        max_count: int = 2000,
        b_range: tuple = (10, 2),
        decay_start: float = 0.0,
        warmup: float = 0.0,
        p: float = 2.0,
    ):
        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.count = 0
        # self.pbar = tqdm(total=max_count)
        self.pbar = tqdm(
            total=max_count,
            desc='Reconstructing Block: Loss (X.X) b (X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
        )
        self.temp_decay = LinearTempDecay(
            max_count,
            rel_start_decay=warmup + (1 - warmup) * decay_start,
            start_b=b_range[0],
            end_b=b_range[1],
        )

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        Parameters
        ----------
        pred: output from quantized model
        tgt: output from FP model
        grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError(
                'Not supported reconstruction loss function: {}'.format(
                    self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    round_vals = module.weight_quantizer.get_soft_targets()
                    round_loss += (self.weight * (1 - (
                        (round_vals - 0.5).abs() * 2).pow(b)).sum())
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 100 == 0:
            self.pbar.set_description(
                'Reconstructing Block: Loss ({:.3f}) b ({:.1f})'.format(
                    float(total_loss), b))
        #     self.pbar.set_postfix(loss=float(total_loss), b=b)
        self.pbar.update(1)
        return total_loss


def block_reconstruction(
    model,
    block: BaseQuantBlock,
    cali_data: torch.Tensor,
    batch_size: int = 32,
    iters: int = 20000,
    weight: float = 0.01,
    opt_mode: str = 'mse',
    asym: bool = False,
    include_act_func: bool = True,
    b_range: tuple = (20, 2),
    warmup: float = 0.0,
    act_quant: bool = False,
    lr: float = 4e-5,
    p: float = 2.0,
    multi_gpu: bool = False,
    optim='adam',
):
    """Block reconstruction to optimize the output from each block.

    Parameters
    ----------
    model: QuantModel
    block: BaseQuantBlock that needs to be optimized
    cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    batch_size: mini-batch size for reconstruction
    iters: optimization iterations for reconstruction,
    weight: the weight of rounding regularization term
    opt_mode: optimization mode
    asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    include_act_func: optimize the output after activation function
    b_range: temperature range
    warmup: proportion of iterations that no scheduling for temperature
    act_quant: use activation quantization or not.
    lr: learning rate for act delta learning
    p: L_p norm minimization
    multi_gpu: use multi-GPU or not, if enabled, we should sync the gradients
    """
    model.set_quant_state(False, False)
    block.set_quant_state(True, act_quant)
    round_mode = 'learned_hard_sigmoid'

    if not include_act_func:
        org_act_func = block.activation_function
        block.activation_function = StraightThrough()

    if not act_quant:
        # Replace weight quantizer to AdaRoundQuantizer
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                module.weight_quantizer = AdaRoundQuantizer(
                    uaq=module.weight_quantizer,
                    round_mode=round_mode,
                    weight_tensor=module.org_weight.data,
                )
                module.weight_quantizer.soft_targets = True

        # Set up optimizer
        opt_params = []
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                opt_params += [module.weight_quantizer.alpha]
        optimizer = optimizer_map[optim](opt_params)
        scheduler = None
    else:
        # Use UniformAffineQuantizer to learn delta
        if hasattr(block.act_quantizer, 'delta'):
            opt_params = [block.act_quantizer.delta]
        else:
            opt_params = []
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                if module.act_quantizer.delta is not None:
                    opt_params += [module.act_quantizer.delta]
        optimizer = optimizer_map[optim](opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=iters,
                                                               eta_min=0.0)

    loss_mode = 'none' if act_quant else 'relaxation'
    rec_loss = opt_mode

    loss_func = BlockLossFunction(
        block,
        round_loss=loss_mode,
        weight=weight,
        max_count=iters,
        rec_loss=rec_loss,
        b_range=b_range,
        decay_start=0,
        warmup=warmup,
        p=p,
    )

    # Save data before optimizing the rounding
    cached_inps, cached_outs = save_inp_oup_data(model, block, cali_data, asym,
                                                 act_quant, batch_size)
    if opt_mode != 'mse':
        cached_grads = save_grad_data(model,
                                      block,
                                      cali_data,
                                      act_quant,
                                      batch_size=batch_size)
    else:
        cached_grads = None
    device = 'cuda'
    for i in range(iters):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]
        cur_inp = cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        cur_grad = cached_grads[idx].to(device) if opt_mode != 'mse' else None

        optimizer.zero_grad()
        out_quant = block(cur_inp)

        err = loss_func(out_quant, cur_out, cur_grad)
        err.backward(retain_graph=True)
        if multi_gpu:
            for p in opt_params:
                dist.all_reduce(p.grad)
        optimizer.step()
        if scheduler:
            scheduler.step()

    torch.cuda.empty_cache()

    # Finish optimization, use hard rounding.
    for name, module in block.named_modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.soft_targets = False

    # Reset original activation function
    if not include_act_func:
        block.activation_function = org_act_func
