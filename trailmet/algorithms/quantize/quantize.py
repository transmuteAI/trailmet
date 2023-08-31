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

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Union
from trailmet.models.resnet import BasicBlock, Bottleneck
from trailmet.models.mobilenet import InvertedResidual
from trailmet.algorithms.quantize.utils import StopForwardException, DataSaverHook, GradSaverHook
from trailmet.algorithms.quantize.utils import LinearTempDecay, Node, GraphPlotter
from trailmet.algorithms.quantize.modules import StraightThrough, QuantModule, BaseQuantBlock
from trailmet.algorithms.quantize.modules import QuantBasicBlock, QuantBottleneck, QuantInvertedResidual
from trailmet.algorithms.algorithms import BaseAlgorithm


supported = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    InvertedResidual: QuantInvertedResidual,
}

class BaseQuantModel(nn.Module):
    """base model wrapping class for quantization algorithms"""
    def __init__(self, model: nn.Module, weight_quant_params: dict = {},
            act_quant_params: dict = {}, fold_bn = True):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        if fold_bn:
            self.model.eval()
            self.search_fold_remove_bn(self.model)
        self.quant_module_refactor(self.model)
        self.quant_modules = [m for m in self.model.modules() if isinstance(m, QuantModule)]


    def search_fold_remove_bn(self, module: nn.Module):
        """
        Recursively search for BatchNorm layers, fold them into the previous 
        Conv2d or Linear layers and set them as a StraightThrough layer.
        """
        prev_module = None
        for name, child_module in module.named_children():
            if self._is_bn(child_module) and self._is_absorbing(prev_module):
                self._fold_bn_into_conv(prev_module, child_module)
                setattr(module, name, StraightThrough())
            elif self._is_absorbing(child_module):
                prev_module = child_module
            else:
                prev_module = self.search_fold_remove_bn(child_module)
        return prev_module

    def quant_module_refactor(self, module: nn.Module):
        """
        Recursively replace Conv2d and Linear layers with QuantModule and other 
        supported network blocks to their respective wrappers, to enable weight 
        and activations quantization.
        """
        prev_quant_module: QuantModule = None
        for name, child_module in module.named_children():
            if type(child_module) in supported:
                setattr(module, name, supported[type(child_module)](
                    child_module, self.weight_quant_params, self.act_quant_params
                ))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(
                    child_module, self.weight_quant_params, self.act_quant_params
                ))
                prev_quant_module = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quant_module is not None:
                    prev_quant_module.activation_function = child_module
                else:
                    continue
            elif isinstance(child_module, StraightThrough):
                continue
            else:
                self.quant_module_refactor(child_module)
            

    def set_quant_state(self, weight_quant: bool = True, act_quant: bool = True):
        """
        :param weight_quant: set True to enable weight quantization
        :param act_quant: set True to enable activation quantization
        """
        for module in self.model.modules():
            if isinstance(module, (QuantModule, BaseQuantBlock)):
                module.set_quant_state(weight_quant, act_quant)

    def quantize_model_till(self, layer, act_quant: bool = False):
        """
        :param layer: layer upto which model is to be quantized.
        :param act_quant: set True for activation quantization
        """
        self.set_quant_state(False, False)
        for name, module in self.model.named_modules():
            if isinstance(module, (QuantModule, BaseQuantBlock)):
                module.set_quant_state(True, act_quant)
            if module == layer:
                break 

    def set_layer_precision(self, weight_bits: list, act_bit: int):
        """
        :param weight_bits: list of bitwidths for layer weights
        :param act_bit: bitwidth for activations
        """
        assert len(weight_bits)==len(self.quant_modules)
        for idx, module in  enumerate(self.quant_modules):
            module.weight_quantizer.bitwidth_refactor(weight_bits[idx])
            if module is not self.quant_modules[-1]:
                module.act_quantizer.bitwidth_refactor(act_bit)

    def forward(self, input):
        return self.model(input)
    
    def _is_bn(self, module):
        return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
    
    def _is_absorbing(self, module):
        return isinstance(module, (nn.Conv2d, nn.Linear))
    
    def _fold_bn_into_conv(self, conv_module, bn_module):
        w, b = self._get_folded_params(conv_module, bn_module)
        if conv_module.bias is None:
            conv_module.bias = nn.Parameter(b)
        else:
            conv_module.bias.data = b
        conv_module.weight.data = w
        bn_module.running_mean = bn_module.bias.data
        bn_module.running_var = bn_module.weight.data ** 2

    def _get_folded_params(self, conv_module, bn_module):
        w = conv_module.weight.data
        y_mean = bn_module.running_mean
        y_var = bn_module.running_var
        safe_std = torch.sqrt(y_var + bn_module.eps)
        w_view = (conv_module.out_channels, 1, 1, 1)
        if bn_module.affine:
            weight = w * (bn_module.weight / safe_std).view(w_view)
            beta = bn_module.bias - bn_module.weight * y_mean / safe_std
            if conv_module.bias is not None:
                bias = bn_module.weight * conv_module.bias / safe_std + beta
            else:
                bias = beta
        else:
            weight = w / safe_std.view(w_view)
            beta = -y_mean / safe_std
            if conv_module.bias is not None:
                bias = conv_module.bias / safe_std + beta
            else:
                bias = beta
        return weight, bias

class BaseQuantization(BaseAlgorithm):
    """base class for quantization algorithms"""
    def __init__(self, **kwargs):
        super(BaseQuantization, self).__init__(**kwargs)
        pass

    def quantize(self, model, dataloaders, method, **kwargs):
        pass

    def get_calib_samples(self, train_loader, num_samples):
        """
        Get calibration-dataset samples for finetuning quantized weights and 
        quantization parameters
        """
        calib_data = []
        for batch in train_loader:
            calib_data.append(batch[0])
            if len(calib_data)*batch[0].size(0) >= num_samples:
                break
        return torch.cat(calib_data, dim=0)[:num_samples]


    def sensitivity_analysis(self, qmodel: BaseQuantModel, dataloader, test_bits, 
            budget, save_path, exp_name):
        qmodel.set_quant_state(False, False)
        inputs = None
        fp_outputs = None
        with torch.no_grad():
            for batch_idx, (inputs, outputs) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                fp_outputs = qmodel(inputs)
                fp_outputs = F.softmax(fp_outputs, dim=1)
                break
        sensitivities = [[0 for i in range(len(qmodel.quant_modules))] 
            for j in range(len(test_bits))]
        for i, layer in enumerate(qmodel.quant_modules):
            for j, bit in enumerate(test_bits):
                layer.set_quant_state(True, True)
                layer.weight_quantizer.bitwidth_refactor(bit)
                layer.weight_quantizer.inited = False
                layer.weight_quantizer.scale_method = 'max'
                with torch.no_grad():
                    tmp_outputs = qmodel(inputs)
                    tmp_outputs = F.softmax(tmp_outputs, dim=1)
                    kld = (F.kl_div(tmp_outputs, fp_outputs, reduction='batchmean') + 
                           F.kl_div(fp_outputs, tmp_outputs, reduction='batchmean')) / 2
                sensitivities[j][i] = kld.item()
                layer.set_quant_state(False, False)
                layer.weight_quantizer.scale_method = 'mse'
        
        gp = GraphPlotter(save_path+'/logs/plots')
        gp.line_plotter(sensitivities, test_bits, '{} bit', f'{exp_name}_layer_sensitivity',
            'layer', 'sensitivity', 'log')

        weight_numels = [qmodule.weight.numel() for qmodule in qmodel.quant_modules]
        node_list = self.dp_most_profit_over_cost(sensitivities, len(qmodel.quant_modules), weight_numels, test_bits)
        constraint = sum(weight_numels)*32*budget / (8*1024*1024)
        good_nodes = [node for node in node_list if node.cost <= constraint]
        bits = []
        node = good_nodes[-1]
        while(node is not None):
            bits.append(node.bit)
            node = node.parent
        bits.reverse()
        bits = bits[1:]
        assert len(bits)==len(qmodel.quant_modules)
        gp.line_plotter([bits], ['weight bits'], title=f'{exp_name}_layer_precisions',
            xlabel='layer', ylabel='bits')
        qmodel_size = 0
        for i, layer in enumerate(qmodel.quant_modules):
            qmodel_size += layer.weight.numel()*bits[i]/(8*1024*1024)
        return bits, qmodel_size, constraint

    def dp_most_profit_over_cost(self, sensitivities, num_layers, weight_numels, bits, constraint=100):
        cost = bits
        profits = []
        for line in sensitivities:
            profits.append([-i for i in line])
        root = Node(cost=0, profit=0, parent=None)
        current_list = [root]
        for layer_id in range(num_layers):
            next_list = []
            for n in current_list:
                n.left = Node(n.cost + cost[0]*weight_numels[layer_id]/(8*1024*1024), 
                                n.profit + profits[0][layer_id],
                                bit = bits[0], parent=n, position='left')
                n.middle = Node(n.cost + cost[1]*weight_numels[layer_id]/(8*1024*1024), 
                                n.profit + profits[1][layer_id],
                                bit = bits[1], parent=n, position='middle')
                n.right = Node(n.cost + cost[2]*weight_numels[layer_id]/(8*1024*1024), 
                                n.profit + profits[2][layer_id],
                                bit = bits[2], parent=n, position='right')
                next_list.extend([n.left, n.middle, n.right])
            next_list.sort(key=lambda x: x.cost, reverse=False)
            pruned_list = []
            for node in next_list:
                if (len(pruned_list)==0 or pruned_list[-1].profit < node.profit) and node.cost <= constraint:
                    pruned_list.append(node)
                else:
                    node.parent.__dict__[node.position] = None
            current_list = pruned_list
        return current_list
    

class BaseQuantLoss:
    def __init__(self, module: Union[QuantModule, BaseQuantBlock], 
            round_loss: str = 'relaxation', weight: float = 1., rec_loss: str = 'mse',
            max_count: int = 2000, b_range: tuple = (10, 2), decay_start: float = 0.0,
            warmup: float = 0.0, p: float = 2.):
        
        self.module = module
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.count = 0
        self.pbar = tqdm(total=max_count)
        self.temp_decay = LinearTempDecay(max_count, 
            rel_start_decay=warmup + (1 - warmup) * decay_start,
            start_b=b_range[0], end_b=b_range[1])

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy
        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = self.lp_norm(pred, tgt, self.p, reduction='none')
        elif self.rec_loss == 'fisher_diag':
            rec_loss = self.fisher_diag(pred, tgt, grad)
        elif self.rec_loss == 'fisher_full':
            rec_loss = self.fisher_full(pred, tgt, grad)
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            if isinstance(self.module, QuantModule):
                round_vals = self.module.weight_quantizer.get_soft_targets()
                round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
            if isinstance(self.module, BaseQuantBlock):
                for name, submodule in self.module.named_modules():
                    if isinstance(submodule, QuantModule):
                        round_vals = submodule.weight_quantizer.get_soft_targets()
                        round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 100 == 0:
            self.pbar.set_postfix(loss=float(total_loss), b=b)
        self.pbar.update(1)
        return total_loss
    
    @staticmethod
    def lp_norm(pred, tgt, p=2.0, reduction = 'mean'):
        if reduction == 'mean':
            return (pred-tgt).abs().pow(p).mean()
        elif reduction == 'none': 
            return (pred-tgt).abs().pow(p).sum(1).mean()
        else:
            raise KeyError
        
    @staticmethod
    def fisher_diag(pred, tgt, grad):
        return ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()

    @staticmethod
    def fisher_full(pred, tgt, grad):
        a = (pred - tgt).abs()
        grad = grad.abs()
        batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
        return  (batch_dotprod * a * grad).mean() / 100
    
class GetLayerInpOut:
    """
    Get the input and output of a specified layer in a quantized model.

    :param model: quantized model for which the input and output needs to be extracted.
    :param layer: the layer for which input and output needs to be extracted.
    :param device: the device on which the computation needs to be performed.
    :param asym: save quantized input and full precision output. [default=False]
    :param act_quant: use activation quantization. [default=False]
    """
    def __init__(self, model: BaseQuantModel, layer: Union[QuantModule, BaseQuantBlock],
            device: torch.device, asym: bool = False, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = device
        self.act_quant = act_quant
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, model_input):
        """
        :param model_input: calibration data samples
        :return: tuple of layer input and output
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
                self.model.set_quant_state(weight_quant=True, act_quant=self.act_quant)
                try:
                    _ = self.model(model_input.to(self.device))
                except StopForwardException:
                    pass
                self.data_saver.store_output = True

        handle.remove()

        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()

        return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach()

class GetLayerGrad:
    """
    Get the gradient a specified layer in a quantized model.

    :param model: quantized model for which the input and output needs to be extracted.
    :param layer: the layer for which input and output needs to be extracted.
    :param device: the device on which the computation needs to be performed.
    :param asym: if True, save quantized input and full precision output. [default=False]
    :param act_quant: use activation quantization. [default=False]
    """
    def __init__(self, model: BaseQuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.device = device
        self.act_quant = act_quant
        self.data_saver = GradSaverHook(True)

    def __call__(self, model_input):
        """
        Compute the gradients of layer output, note that we compute the
        gradient by calculating the KL loss between fp model and quant model

        :param model_input: calibration data samples
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
                loss = F.kl_div(F.log_softmax(out_q, dim=1), F.softmax(out_fp, dim=1), reduction='batchmean')
                loss.backward()
            except StopForwardException:
                pass

        handle.remove()
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()
        return self.data_saver.grad_out.data