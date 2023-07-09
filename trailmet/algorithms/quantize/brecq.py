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
import torch.distributed as dist
from trailmet.utils import seed_everything
from trailmet.algorithms.quantize.quantize import (
    BaseQuantization,
    FoldBN,
    StraightThrough,
)
from trailmet.models.resnet import BasicBlock, Bottleneck
from trailmet.models.mobilenet import InvertedResidual
from trailmet.algorithms.quantize.qmodel import (
    QuantBasicBlock,
    QuantBottleneck,
    QuantInvertedResidual,
    QuantModule,
    BaseQuantBlock,
)
from trailmet.algorithms.quantize.reconstruct import (
    layer_reconstruction,
    block_reconstruction,
)

import logging
from datetime import datetime
from tqdm import tqdm
import wandb
import pandas as pd
import numpy as np
import os
import time

from trailmet.utils import AverageMeter, accuracy, save_checkpoint

logger = logging.getLogger(__name__)

supported = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    InvertedResidual: QuantInvertedResidual,
}


class BRECQ(BaseQuantization):
    """
    Class for post-training quantization using block reconstruction method
    based on - BRECQ: PUSHING THE LIMIT OF POST-TRAINING QUANTIZATION
    BY BLOCK RECONSTRUCTION [https://arxiv.org/abs/2102.05426]

    Parameters
    ----------
    model (nn.Module): Model to be used
    dataloaders (dict): Dictionary with dataloaders for train, test, val
    W_BITS: bitwidth for weight quantization
    A_BITS: bitwidth for activation quantization
    CHANNEL_WISE: apply channel_wise quantization for weights
    ACT_QUANT: apply activation quantization
    SET_8BIT_HEAD_STEM: Set the first and the last layer to 8-bit
    NUM_SAMPLES: size of calibration dataset
    WEIGHT: weight of rounding cost vs the reconstruction loss
    ITERS_W: number of iteration for AdaRound
    ITERS_A: number of iteration for LSQ
    LR: learning rate for LSQ
    """

    def __init__(self, model: nn.Module, dataloaders, **kwargs):
        super(BRECQ, self).__init__(**kwargs)
        self.model = copy.deepcopy(model)
        self.train_loader = dataloaders['train']
        self.test_loader = dataloaders['test']
        self.kwargs = kwargs
        self.w_bits = self.kwargs.get('W_BITS', 8)
        self.a_bits = self.kwargs.get('A_BITS', 8)
        self.channel_wise = self.kwargs.get('CHANNEL_WISE', True)
        self.act_quant = self.kwargs.get('ACT_QUANT', True)
        self.set_8bit_head_stem = self.kwargs.get('SET_8BIT_HEAD_STEM', False)
        self.precision_config = self.kwargs.get('PREC_CONFIG', [])
        self.num_samples = self.kwargs.get('NUM_SAMPLES', 1024)
        self.weight = self.kwargs.get('WEIGHT', 0.01)
        self.iters_w = self.kwargs.get('ITERS_W', 10000)
        self.iters_a = self.kwargs.get('ITERS_A', 10000)
        self.optimizer = self.kwargs.get('OPTIMIZER', 'adam')
        self.lr = self.kwargs.get('LR', 4e-4)
        self.gpu_id = self.kwargs.get('GPU_ID', 0)
        self.calib_bs = self.kwargs.get('CALIB_BS', 64)
        self.seed = self.kwargs.get('SEED', 42)
        self.p = 2.4  # Lp norm minimization for LSQ
        self.b_start = 20  # temperature at the beginning of calibration
        self.b_end = 2  # temperature at the end of calibration
        self.test_before_calibration = True
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        torch.cuda.set_device(self.gpu_id)
        seed_everything(self.seed)
        print('==> Using seed :', self.seed)

        self.wandb_monitor = self.kwargs.get('WANDB', 'False')
        self.dataset_name = dataloaders['train'].dataset.__class__.__name__
        self.save = './checkpoints/'

        self.name = '_'.join([
            self.dataset_name,
            f'{self.a_bits}',
            f'{self.lr}',
            datetime.now().strftime('%b-%d_%H:%M:%S'),
        ])

        os.makedirs(f'{os.getcwd()}/logs/BRECQ', exist_ok=True)
        os.makedirs(self.save, exist_ok=True)
        self.logger_file = f'{os.getcwd()}/logs/BRECQ/{self.name}.log'

        logging.basicConfig(
            filename=self.logger_file,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )

        logger.info(f'Experiment Arguments: {self.kwargs}')

        if self.wandb_monitor:
            wandb.init(project='Trailmet BRECQ', name=self.name)
            wandb.config.update(self.kwargs)

    def compress_model(self):
        """Method to build quantization parameters and finetune weights and/or
        activations."""
        wq_params = {
            'n_bits': self.w_bits,
            'channel_wise': self.channel_wise,
            'scale_method': 'mse',
        }
        aq_params = {
            'n_bits': self.a_bits,
            'channel_wise': False,
            'scale_method': 'mse',
            'leaf_param': self.act_quant,
        }
        self.model = self.model.to(self.device)
        self.model.eval()
        self.qnn = QuantModel(model=self.model,
                              weight_quant_params=wq_params,
                              act_quant_params=aq_params)
        self.qnn = self.qnn.to(self.device)
        self.qnn.eval()

        for i in range(len(self.precision_config)):
            conf = self.precision_config[i]
            self.qnn.set_layer_precision(conf[2], conf[3], conf[0], conf[1])
            print(
                f'==> Layers from {conf[0]} to {conf[1]} set to precision w{conf[2]}a{conf[3]}'
            )
            logger.info(
                f'==> Layers from {conf[0]} to {conf[1]} set to precision w{conf[2]}a{conf[3]}'
            )

        if self.set_8bit_head_stem:
            print('==> Setting the first and the last layer to 8-bit')
            logger.info('==> Setting the first and the last layer to 8-bit')
            self.qnn.set_first_last_layer_to_8bit()

        self.cali_data = self.get_calib_samples(self.train_loader,
                                                self.num_samples)
        # device = next(self.qnn.parameters()).device

        # Initialize weight quantization parameters
        self.qnn.set_quant_state(True, False)
        print('==> Initializing weight quantization parameters')
        logger.info('==> Initializing weight quantization parameters')
        _ = self.qnn(self.cali_data[:self.calib_bs].to(self.device))
        if self.test_before_calibration:
            valid_loss, valid_top1_acc, valid_top5_acc = self.test(
                self.qnn, self.test_loader, nn.CrossEntropyLoss())
            print('Quantized accuracy before brecq: {}'.format(valid_top1_acc))
            logger.info(
                'Quantized accuracy before brecq: {}'.format(valid_top1_acc))

        # Start weight calibration
        kwargs = dict(
            cali_data=self.cali_data,
            iters=self.iters_w,
            weight=self.weight,
            asym=True,
            b_range=(self.b_start, self.b_end),
            warmup=0.2,
            act_quant=False,
            opt_mode='mse',
            optim=self.optimizer,
        )
        print('==> Starting weight calibration')
        logger.info('==> Starting weight calibration')
        self.reconstruct_model(self.qnn, **kwargs)
        self.qnn.set_quant_state(weight_quant=True, act_quant=False)
        valid_loss, valid_top1_acc, valid_top5_acc = self.test(
            self.qnn, self.test_loader, nn.CrossEntropyLoss())
        print('Weight quantization accuracy: {}'.format(valid_top1_acc))

        if self.act_quant:
            # Initialize activation quantization parameters
            self.qnn.set_quant_state(True, True)
            with torch.no_grad():
                _ = self.qnn(self.cali_data[:self.calib_bs].to(self.device))

            # Disable output quantization because network output
            # does not get involved in further computation
            self.qnn.disable_network_output_quantization()

            # Start activation rounding calibration
            kwargs = dict(
                cali_data=self.cali_data,
                iters=self.iters_a,
                act_quant=True,
                opt_mode='mse',
                lr=self.lr,
                p=self.p,
                optim=self.optimizer,
            )
            self.reconstruct_model(self.qnn, **kwargs)
            self.qnn.set_quant_state(weight_quant=True, act_quant=True)
            valid_loss, valid_top1_acc, valid_top5_acc = self.test(
                self.qnn, self.test_loader, nn.CrossEntropyLoss())
            print('Full quantization (W{}A{}) accuracy: {}'.format(
                self.w_bits, self.a_bits, valid_top1_acc))
            logger.info('Full quantization (W{}A{}) accuracy: {}'.format(
                self.w_bits, self.a_bits, valid_top1_acc))

        return self.qnn

    def reconstruct_model(self, model: nn.Module, **kwargs):
        """Method for model parameters reconstruction.

        Takes in quantized model and optimizes weights by applying layer-wise
        reconstruction for first and last layer, and block reconstruction
        otherwise.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    logger.info(
                        'Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    logger.info('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(self.qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    logger.info(
                        'Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    logger.info('Reconstruction for block {}'.format(name))
                    block_reconstruction(self.qnn, module, **kwargs)
            else:
                self.reconstruct_model(module, **kwargs)

    def test(self, model, dataloader, loss_fn):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        epoch_iterator = tqdm(
            dataloader,
            desc=
            'Validating network (X / X Steps) (batch time=X.Xs) (loss=X.X) (top1=X.X) (top5=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        model.eval()
        model.to(self.device)

        with torch.no_grad():
            end = time.time()

            for i, (images, labels) in enumerate(epoch_iterator):
                images = images.to(self.device, dtype=torch.float)
                labels = labels.to(self.device)

                preds = model(images)

                loss = loss_fn(preds, labels)

                pred1, pred5 = accuracy(preds, labels, topk=(1, 5))

                n = images.size(0)
                losses.update(loss.item(), n)
                top1.update(pred1[0], n)
                top5.update(pred5[0], n)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                epoch_iterator.set_description(
                    'Validating network (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
                    % (
                        (i + 1),
                        len(dataloader),
                        batch_time.val,
                        losses.val,
                        top1.val,
                        top5.val,
                    ))

                logger.info(
                    'Validating network (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
                    % (
                        (i + 1),
                        len(dataloader),
                        batch_time.val,
                        losses.val,
                        top1.val,
                        top5.val,
                    ))

                if self.wandb_monitor:
                    wandb.log({
                        'val_loss': losses.val,
                        'val_top1_acc': top1.val,
                        'val_top5_acc': top5.val,
                    })

            print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'.format(
                top1=top1, top5=top5))
        return losses.avg, top1.avg, top5.avg


class QuantModel(nn.Module):
    """Recursively replace the normal conv2d and Linear layer to QuantModule,
    to enable calculating activation statistics and storing scaling factors.

    Parameters
    ----------
    model (nn.Module): nn.Module with nn.Conv2d or nn.Linear in its children
    weight_quant_params (dict): quantization parameters like n_bits for weight
        quantizer
    act_quant_params(dict): quantization parameters like n_bits for activation
        quantizer
    """

    def __init__(
        self,
        model: nn.Module,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
    ):
        super().__init__()
        self.model = model
        bn = FoldBN()
        bn.search_fold_and_remove_bn(self.model)
        self.quant_module_refactor(self.model, weight_quant_params,
                                   act_quant_params)
        self.quant_modules = [
            m for m in self.model.modules() if isinstance(m, QuantModule)
        ]

    def quant_module_refactor(
        self,
        module: nn.Module,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
    ):
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if type(child_module) in supported:
                setattr(
                    module,
                    name,
                    supported[type(child_module)](child_module,
                                                  weight_quant_params,
                                                  act_quant_params),
                )

            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(
                    module,
                    name,
                    QuantModule(child_module, weight_quant_params,
                                act_quant_params),
                )
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params,
                                           act_quant_params)

    def set_quant_state(self,
                        weight_quant: bool = False,
                        act_quant: bool = False):
        """
        :param weight_quant: set True for weight quantization
        :param act_quant: set True for activation quantization
        """
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def quantize_model_till(self, layer, act_quant: bool = False):
        """
        :param layer: block/layer upto which model is to be quantized.
        :param act_quant: set True for activation quantization
        """
        self.set_quant_state(False, False)
        for name, module in self.model.named_modules():
            if isinstance(module, (QuantModule, BaseQuantBlock)):
                module.set_quant_state(True, act_quant)
            if module == layer:
                break

    def forward(self, input):
        return self.model(input)

    def set_first_last_layer_to_8bit(self):
        """Set the precision (bitwidth) used for quantizing weights and
        activations to 8-bit for the first and last layers of the model.

        Also ignore reconstruction for the first layer.
        """
        assert (len(self.quant_modules)
                >= 2), 'Model has less than 2 quantization modules'
        self.quant_modules[0].weight_quantizer.bitwidth_refactor(8)
        self.quant_modules[0].act_quantizer.bitwidth_refactor(8)
        self.quant_modules[-1].weight_quantizer.bitwidth_refactor(8)
        self.quant_modules[-2].act_quantizer.bitwidth_refactor(8)
        self.quant_modules[0].ignore_reconstruction = True

    def disable_network_output_quantization(self):
        self.quant_modules[-1].disable_act_quant = True

    def set_layer_precision(self, weight_bit=8, act_bit=8, start=0, end=None):
        """Set the precision (bitwidth) used for quantizing weights and
        activations for a range of layers in the model.

        :param weight_bit: number of bits to use for quantizing weights
        :param act_bit: number of bits to use for quantizing activations
        :param start: index of the first layer to set the precision for
            (default: 0)
        :param end: index of the last layer to set the precision for (default:
            None, i.e., the last layer)
        """
        assert start >= 0 and end >= 0, 'layer index cannot be negative'
        assert start < len(self.quant_modules) and end < len(
            self.quant_modules), 'layer index out of range'

        for module in self.quant_modules[start:end + 1]:
            module.weight_quantizer.bitwidth_refactor(weight_bit)
            if module is not self.quant_modules[-1]:
                module.act_quantizer.bitwidth_refactor(act_bit)

    def synchorize_activation_statistics(self):
        """Synchronize the statistics of the activation quantizers across all
        distributed workers."""
        for m in self.modules():
            if isinstance(m, QuantModule):
                if m.act_quantizer.delta is not None:
                    m.act_quantizer.delta.data /= dist.get_world_size()
                    dist.all_reduce(m.act_quantizer.delta.data)
