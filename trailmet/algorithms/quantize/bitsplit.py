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
import copy, random, pickle
import torch
import torch.nn as nn
from collections import OrderedDict
from trailmet.utils import seed_everything
from trailmet.algorithms.quantize.quantize import BaseQuantization
from trailmet.models.resnet import BasicBlock, Bottleneck
from trailmet.models.mobilenet import InvertedResidual
from trailmet.algorithms.quantize.qmodel import (
    QBasicBlock,
    QBottleneck,
    QInvertedResidual,
)
from trailmet.algorithms.quantize.methods import BitSplitQuantizer, ActQuantizer

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

global feat, prev_feat, conv_feat


def hook(module, input, output):
    global feat
    feat = output.data.cpu().numpy()


def current_input_hook(module, inputdata, outputdata):
    global prev_feat
    prev_feat = inputdata[0].data


def conv_hook(module, inputdata, outputdata):
    global conv_feat
    conv_feat = outputdata.data


class QuantModel(nn.Module):
    """
    Parameters
    ----------
        model (nn.Module): Model to be used.
        arch (str): Architecture to be used.
    """

    def __init__(self, model: nn.Module, arch='ResNet50'):
        super().__init__()
        self.supported = {
            BasicBlock: QBasicBlock,
            Bottleneck: QBottleneck,
            InvertedResidual: QInvertedResidual,
        }
        if arch == 'ResNet50':
            setattr(model, 'quant', ActQuantizer())
            setattr(model, 'fc', nn.Sequential(ActQuantizer(), model.fc))
        if arch == 'MobileNetV2':
            setattr(model, 'conv2', nn.Sequential(ActQuantizer(), model.conv2))
            setattr(model, 'linear', nn.Sequential(ActQuantizer(),
                                                   model.linear))
        self.quant_block_refactor(model)

    def quant_block_refactor(self, module: nn.Module):
        """Recursively modify the supported conv-blocks to add activation
        quantization layers :param module: nn.Module with supported conv-block
        classes in its children."""
        for name, child_module in module.named_children():
            if type(child_module) in self.supported:
                setattr(module, name,
                        self.supported[type(child_module)](child_module))
            elif isinstance(child_module,
                            (nn.Conv2d, nn.Linear, nn.ReLU, nn.ReLU6)):
                continue
            else:
                self.quant_block_refactor(child_module)


class BitSplit(BaseQuantization):
    """
    Class for post-training quantization using bit-split and stitching method
    based on - Towards accurate post-training network quantization via
    bit-split and stitching [https://dl.acm.org/doi/abs/10.5555/3524938.3525851]

    Parameters
    ----------
    model (nn.Module): Model to be used
    dataloaders (dict): Dictionary with dataloaders for train, test, val
    W_BITS: bitwidth for weight quantization
    A_BITS: bitwidth for activation quantization
    CHANNEL_WISE: apply channel-wise quantization for weights
    ACT_QUANT: apply activation quantization
    HEAD_STEM_PRECISION: bitwidth for first and last layer
    PREC_CONFIG: list of bitwidths of the body for mixed precision
    CALIB_BATCHES: num of batches in calibration dataset
    LOAD_ACT_SCALES: load precomputed weight scales
    LOAD_WEIGHT_SCALES: load precomputed activation scales
    SAVE_PATH: path for storing quantized weights and scales
    """

    def __init__(self, model: nn.Module, dataloaders, **kwargs):
        super(BitSplit, self).__init__(**kwargs)
        self.model = model
        self.train_loader = dataloaders['train']
        self.kwargs = kwargs
        self.w_bits = self.kwargs.get('W_BITS', 8)
        self.a_bits = self.kwargs.get('A_BITS', 8)
        self.gpu_id = self.kwargs.get('GPU_ID', 0)
        self.seed = self.kwargs.get('SEED', 42)
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        torch.cuda.set_device(self.gpu_id)
        seed_everything(self.seed)
        self.save_path = self.kwargs.get('SAVE_PATH', './')
        self.arch = self.kwargs.get('ARCH', '')
        self.dataset = self.kwargs.get('DATASET', '')
        self.precision_config = self.kwargs.get('PREC_CONFIG', [])
        if self.precision_config:
            w_prefix = str(self.precision_config[0]) + '_mix'
        else:
            w_prefix = str(self.w_bits)
        self.prefix = self.save_path + self.arch + '_' + self.dataset + '/W' + w_prefix
        if not os.path.exists(self.prefix):
            os.makedirs(self.prefix)
        self.load_act_scales = self.kwargs.get('LOAD_ACT_SCALES', False)
        self.load_weight_scales = self.kwargs.get('LOAD_WEIGHT_SCALES', False)
        self.calib_batches = self.kwargs.get('CALIB_BATCHES', 8)
        self.act_quant = self.kwargs.get('ACT_QUANT', True)
        self.head_stem_precision = self.kwargs.get('HEAD_STEM_PRECISION', None)

        self.wandb_monitor = self.kwargs.get('WANDB', 'False')
        self.dataset_name = dataloaders['train'].dataset.__class__.__name__
        self.save = './checkpoints/'

        self.name = '_'.join([
            self.dataset_name,
            str(self.a_bits),
            datetime.now().strftime('%b-%d_%H:%M:%S'),
        ])

        os.makedirs(f'{os.getcwd()}/logs/BitSplit', exist_ok=True)
        os.makedirs(self.save, exist_ok=True)
        self.logger_file = f'{os.getcwd()}/logs/BitSplit/{self.name}.log'

        logging.basicConfig(
            filename=self.logger_file,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )

        logger.info(f'Experiment Arguments: {self.kwargs}')

        if self.wandb_monitor:
            wandb.init(project='Trailmet BitSplit', name=self.name)
            wandb.config.update(self.kwargs)

    def compress_model(self):
        self.model.to(self.device)
        self.qmodel = copy.deepcopy(self.model)
        QuantModel(self.qmodel, self.arch)

        self.act_quant_modules = []
        for m in self.qmodel.modules():
            if isinstance(m, ActQuantizer):
                m.set_bitwidth(self.a_bits)
                self.act_quant_modules.append(m)
        self.act_quant_modules[-1].set_bitwidth(max(8, self.a_bits))
        assert self.arch in ['MobileNetV2', 'ResNet50']

        print('==> Starting weight quantization')
        logger.info('==> Starting weight quantization')
        self.weight_quantizer(load_only=self.load_weight_scales)
        if self.act_quant:
            if self.load_act_scales:
                scales = np.load(self.prefix + '/act_' + str(self.a_bits) +
                                 '_scales.npy')
                for index, q_module in enumerate(self.act_quant_modules):
                    q_module.set_scale(scales[index])
            else:
                print("==> Starting '{}-bit' activation quantization".format(
                    self.a_bits))
                logger.info(
                    "==> Starting '{}-bit' activation quantization".format(
                        self.a_bits))
                self.act_quantizer(self.qmodel,
                                   prefix=self.prefix,
                                   n_batches=self.calib_batches)

        save_checkpoint(
            {
                'state_dict': self.qmodel.module.state_dict(),
            },
            is_best=False,
            save=self.save,
        )
        # save_state_dict(self.qmodel.state_dict(), self.prefix, filename='state_dict.pth')

        print('testing quantized model')
        logger.info('testing quantized model')

        val_top1_acc_list = []
        val_top5_acc_list = []

        valid_loss, valid_top1_acc, valid_top5_acc = self.test(
            self.qmodel, self.dataloaders['val'], nn.CrossEntropyLoss())
        val_top1_acc_list.append(valid_top1_acc.cpu().numpy())
        val_top5_acc_list.append(valid_top5_acc.cpu().numpy())

        df_data = np.array([
            val_top1_acc_list,
            val_top5_acc_list,
        ]).T
        df = pd.DataFrame(
            df_data,
            columns=[
                'Validation Top1',
                'Validation Top5',
            ],
        )
        df.to_csv(
            f'{os.getcwd()}/logs/BitSplit/{self.name}.csv',
            index=False,
        )

    # TODO :  Use functions to process submodules of respective models so that adding new models in future is easier
    def weight_quantizer(self, load_only=False):
        """Find optimum weight quantization scales for ResNet & Mobilenet."""
        #### Quantizer for MobilenetV2 ####
        if self.arch == 'MobileNetV2':
            count = 3
            for i in range(len(self.model.layers)):
                if len(self.model.layers[i].shortcut) > 0:
                    count += 4
                else:
                    count += 3
            pbar = tqdm(total=count)
            layer_to_block = [
                1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4
            ]
            assert (len(self.precision_config) == 0 or len(
                self.precision_config) == 7), 'config list must be of length 7'
            # quantize first conv layer
            conv = self.model.conv1
            conv_quan = self.qmodel.conv1
            w_bit = self.w_bits
            if self.head_stem_precision is not None:
                w_bit = self.head_stem_precision
            if self.precision_config:
                w_bit = self.precision_config[0]
            if w_bit == 32:
                conv_quan.weight.data.copy_(conv.weight.data)
            else:
                if not load_only:
                    conduct_ofwa(
                        self.train_loader,
                        self.model,
                        self.qmodel,
                        conv,
                        conv_quan,
                        w_bit,
                        self.calib_batches,
                        prefix=self.prefix + '/conv1',
                        device=self.device,
                        ec=False,
                    )
                load_ofwa(conv, conv_quan, prefix=self.prefix + '/conv1')
            pbar.update(1)
            time.sleep(0.1)
            # quantize blocks
            for layer_idx in range(len(self.model.layers)):
                current_layer_pretrained = self.model.layers[layer_idx]
                current_layer_quan = self.qmodel.layers[layer_idx]
                w_bit = (self.precision_config[layer_to_block[layer_idx]]
                         if self.precision_config else self.w_bits)
                skip = w_bit == 32
                pkl_path = self.prefix + '/layer' + str(layer_idx)
                # conv layers
                for idx in range(1, 4):
                    conv = eval('current_layer_pretrained.conv{}'.format(idx))
                    conv_quan = eval('current_layer_quan.conv{}'.format(idx))
                    if skip:
                        conv_quan.weight.data.copy_(conv.weight.data)
                    else:
                        if not load_only:
                            conduct_ofwa(
                                self.train_loader,
                                self.model,
                                self.qmodel,
                                conv,
                                conv_quan,
                                w_bit,
                                self.calib_batches,
                                prefix=pkl_path + '_conv{}'.format(idx),
                                device=self.device,
                                dw=(idx == 2),
                                ec=False,
                            )
                        load_ofwa(conv,
                                  conv_quan,
                                  prefix=pkl_path + '_conv' + str(idx))
                    pbar.update(1)
                    time.sleep(0.1)
                # shortcut layer
                if len(current_layer_pretrained.shortcut) > 0:
                    conv = current_layer_pretrained.shortcut[0]
                    conv_quan = current_layer_quan.shortcut[0]
                    if skip:
                        conv_quan.weight.data.copy_(conv.weight.data)
                    else:
                        if not load_only:
                            conduct_ofwa(
                                self.train_loader,
                                self.model,
                                self.qmodel,
                                conv,
                                conv_quan,
                                w_bit,
                                self.calib_batches,
                                prefix=pkl_path + '_shortcut'.format(idx),
                                device=self.device,
                                ec=False,
                            )
                        load_ofwa(conv,
                                  conv_quan,
                                  prefix=pkl_path + '_shortcut')
                    pbar.update(1)
                    time.sleep(0.1)
            # quantize last conv layer
            conv = self.model.conv2
            conv_quan = self.qmodel.conv2[1]
            if self.precision_config:
                w_bit = self.precision_config[-2]
            if w_bit == 32:
                conv_quan.weight.data.copy_(conv.weight.data)
            else:
                if not load_only:
                    conduct_ofwa(
                        self.train_loader,
                        self.model,
                        self.qmodel,
                        conv,
                        conv_quan,
                        w_bit,
                        self.calib_batches,
                        prefix=self.prefix + '/conv2',
                        device=self.device,
                        ec=False,
                    )
                load_ofwa(conv, conv_quan, prefix=self.prefix + '/conv2')
            pbar.update(1)
            time.sleep(0.1)
            # quantize last linear layer
            conv = self.model.linear
            conv_quan = self.qmodel.linear[1]
            w_bit = self.w_bits
            if self.head_stem_precision is not None:
                w_bit = self.head_stem_precision
            if self.precision_config:
                w_bit = self.precision_config[-1]
            if w_bit == 32:
                conv_quan.weight.data.copy_(conv.weight.data)
            else:
                if not load_only:
                    conduct_ofwa(
                        self.train_loader,
                        self.model,
                        self.qmodel,
                        conv,
                        conv_quan,
                        w_bit,
                        self.calib_batches,
                        prefix=self.prefix + '/linear',
                        device=self.device,
                        ec=False,
                    )
                load_ofwa(conv, conv_quan, prefix=self.prefix + '/linear')
            pbar.update(1)
            pbar.close()

        #### Quantizer for Resnet50 ####
        elif self.arch == 'ResNet50':
            count = 2
            for i in range(1, 5):
                layer = eval('self.model.layer{}'.format(i))
                for j in range(len(layer)):
                    count += 3
                    if layer[j].downsample is not None:
                        count += 1
            pbar = tqdm(total=count)
            # quantize first conv layer
            conv = self.model.conv1
            conv_quan = self.qmodel.conv1
            w_bit = self.w_bits
            if self.head_stem_precision is not None:
                w_bit = self.head_stem_precision
            if self.precision_config:
                w_bit = self.precision_config[0]
            if w_bit == 32:
                conv_quan.weight.data.copy_(conv.weight.data)
            else:
                if not load_only:
                    conduct_ofwa(
                        self.train_loader,
                        self.model,
                        self.qmodel,
                        conv,
                        conv_quan,
                        w_bit,
                        self.calib_batches,
                        prefix=self.prefix + '/conv1',
                        device=self.device,
                        ec=False,
                    )
                load_ofwa(conv, conv_quan, prefix=self.prefix + '/conv1')
            pbar.update(1)
            time.sleep(0.1)
            # quantize blocks
            for layer_idx in range(1, 5):
                current_layer_pretrained = eval(
                    'self.model.layer{}'.format(layer_idx))
                current_layer_quan = eval(
                    'self.qmodel.layer{}'.format(layer_idx))
                w_bit = (self.precision_config[layer_idx]
                         if self.precision_config else self.w_bits)
                skip = w_bit == 32
                for block_idx in range(len(current_layer_pretrained)):
                    current_block_pretrained = current_layer_pretrained[
                        block_idx]
                    current_block_quan = current_layer_quan[block_idx]
                    pkl_path = (self.prefix + '/layer' + str(layer_idx) +
                                '_block' + str(block_idx))
                    # conv layers
                    for idx in range(1, 4):
                        conv = eval(
                            'current_block_pretrained.conv{}'.format(idx))
                        conv_quan = eval(
                            'current_block_quan.conv{}'.format(idx))
                        if skip:
                            conv_quan.weight.data.copy_(conv.weight.data)
                        else:
                            if not load_only:
                                conduct_ofwa(
                                    self.train_loader,
                                    self.model,
                                    self.qmodel,
                                    conv,
                                    conv_quan,
                                    w_bit,
                                    self.calib_batches,
                                    prefix=pkl_path + '_conv{}'.format(idx),
                                    device=self.device,
                                    ec=False,
                                )
                            load_ofwa(conv,
                                      conv_quan,
                                      prefix=pkl_path + '_conv' + str(idx))
                        pbar.update(1)
                        time.sleep(0.1)
                    # downsample
                    if current_block_pretrained.downsample is not None:
                        conv = current_block_pretrained.downsample[0]
                        conv_quan = current_block_quan.downsample[0]
                        if skip:
                            conv_quan.weight.data.copy_(conv.weight.data)
                        else:
                            if not load_only:
                                conduct_ofwa(
                                    self.train_loader,
                                    self.model,
                                    self.qmodel,
                                    conv,
                                    conv_quan,
                                    w_bit,
                                    self.calib_batches,
                                    prefix=pkl_path + '_downsample',
                                    device=self.device,
                                    ec=False,
                                )
                            load_ofwa(conv,
                                      conv_quan,
                                      prefix=pkl_path + '_downsample')
                        pbar.update(1)
                        time.sleep(0.1)
            # quantize last fc layer
            conv = self.model.fc
            conv_quan = self.qmodel.fc[1]
            w_bit = self.w_bits
            if self.head_stem_precision is not None:
                w_bit = self.head_stem_precision
            if self.precision_config:
                w_bit = self.precision_config[-1]
            if w_bit == 32:
                conv_quan.weight.data.copy_(conv.weight.data)
            else:
                if not load_only:
                    conduct_ofwa(
                        self.train_loader,
                        self.model,
                        self.qmodel,
                        conv,
                        conv_quan,
                        w_bit,
                        self.calib_batches,
                        prefix=self.prefix + '/fc',
                        device=self.device,
                        ec=False,
                    )
                load_ofwa(conv, conv_quan, prefix=self.prefix + '/fc')
            pbar.update(1)
            pbar.close()
        else:
            raise NotImplementedError

    # TODO : Write this in a more cleaner way
    def act_quantizer(self, model, prefix, n_batches):
        """Find optimum activation quantization scale for ResNet model based on
        feature map."""

        # train_batches = iter(self.train_loader)
        # per_batch = len(next(train_batches)[1])
        # act_sta_len = (n_batches+1)*per_batch
        def get_safe_len(x):
            x /= 10
            y = 1
            while x >= 10:
                x /= 10
                y *= 10
            return int(y)

        act_sta_len = 3000000
        feat_buf = np.zeros(act_sta_len)
        scales = np.zeros(len(self.act_quant_modules))

        pbar = tqdm(
            self.act_quant_modules,
            desc=
            'Activation quantization, q_module [X] (X / X Steps) (prev_layer_scale=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )
        with torch.no_grad():
            for index, q_module in enumerate(pbar):
                batch_iterator = iter(self.train_loader)
                images, targets = next(batch_iterator)
                images = images.cuda()
                targets = targets.cuda()

                handle = q_module.register_forward_hook(hook)
                model(images)
                feat_len = feat.size
                per_batch = min(get_safe_len(feat_len), 100000)
                n_batches = int(act_sta_len / per_batch)

                repeat = True
                while repeat:
                    repeat = False
                    for batch_idx in range(0, n_batches):
                        pbar.set_description(
                            'Activation quantization, q_module [%d] (%d / %d Steps) (prev_layer_scale=%2.5f)'
                            % (
                                index,
                                batch_idx + 1,
                                n_batches,
                                scales[index - 1],
                            ))
                        images, targets = next(batch_iterator)
                        images = images.cuda(device=self.device,
                                             non_blocking=True)
                        model(images)
                        if q_module.signed:
                            feat_tmp = np.abs(feat).reshape(-1)
                        else:
                            feat_tmp = feat[feat > 0].reshape(-1)
                            if feat_tmp.size < per_batch:
                                per_batch = int(per_batch / 10)
                                n_batches = int(n_batches * 10)
                                repeat = True
                                break
                        np.random.shuffle(feat_tmp)
                        feat_buf[batch_idx * per_batch:(batch_idx + 1) *
                                 per_batch] = feat_tmp[0:per_batch]
                    if not repeat:
                        scales[index] = q_module.init_quantization(feat_buf)
                handle.remove()

        pbar.close()
        np.save(
            os.path.join(prefix, 'act_' + str(self.a_bits) + '_scales.npy'),
            scales)
        for index, q_module in enumerate(self.act_quant_modules):
            q_module.set_scale(scales[index])

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


def conduct_ofwa(
    train_loader,
    model_pretrained,
    model_quan,
    conv,
    conv_quan,
    bitwidth,
    n_batches,
    device,
    num_epochs=100,
    prefix=None,
    dw=False,
    ec=False,
):
    # for fc
    if not hasattr(conv, 'kernel_size'):
        W = conv.weight.data  # .cpu()
        W_shape = W.shape
        B_sav, B, alpha = BitSplitQuantizer(W.cpu().numpy(), bitwidth).ofwa()
        # B_sav, B, alpha = ofwa(W.cpu().numpy(), bitwidth)
        with open(prefix + '_fwa.pkl', 'wb') as f:
            pickle.dump({'B': B, 'alpha': alpha}, f, pickle.HIGHEST_PROTOCOL)
        if ec:
            W_r = np.multiply(B, np.expand_dims(alpha, 1)).reshape(W_shape)
            conv_quan.weight.data.copy_(torch.from_numpy(W_r))
        return

    # conv parameters
    kernel_h, kernel_w = conv.kernel_size
    pad_h, pad_w = conv.padding
    stride_h, stride_w = conv.stride

    handle_prev = conv_quan.register_forward_hook(current_input_hook)
    handle_conv = conv.register_forward_hook(conv_hook)

    batch_iterator = iter(train_loader)

    # weights and bias
    W = conv.weight.data  # .cpu()
    if conv.bias is None:
        bias = torch.zeros(W.shape[0]).to(conv.weight.device)
    else:
        bias = conv.bias.data  # .cpu()

    # feat extract
    per_batch = 400
    input, target = next(batch_iterator)
    input_pretrained = input.cuda(device=device, non_blocking=True)
    input_quan = input.cuda(device=device, non_blocking=True)
    model_pretrained(input_pretrained)
    model_quan(input_quan)
    [prev_feat_n, prev_feat_c, prev_feat_h, prev_feat_w] = prev_feat.shape
    [conv_feat_n, conv_feat_c, conv_feat_h, conv_feat_w] = conv_feat.shape

    X = torch.zeros(n_batches * per_batch, prev_feat_c, kernel_h,
                    kernel_w).to(device)
    Y = torch.zeros(n_batches * per_batch, conv_feat_c).to(device)

    for batch_idx in range(0, n_batches):
        input, target = next(batch_iterator)
        input_pretrained = input.cuda(device=device, non_blocking=True)
        model_pretrained(input_pretrained)
        input_quan = input.cuda(device=device, non_blocking=True)
        model_quan(input_quan)

        prev_feat_pad = torch.zeros(prev_feat_n, prev_feat_c,
                                    prev_feat_h + 2 * pad_h,
                                    prev_feat_w + 2 * pad_w).to(device)
        prev_feat_pad[:, :, pad_h:pad_h + prev_feat_h,
                      pad_w:pad_w + prev_feat_w] = prev_feat
        prev_feat_pad = (prev_feat_pad.unfold(2, kernel_h, stride_h).unfold(
            3, kernel_w, stride_w).permute(0, 2, 3, 1, 4, 5))
        [
            feat_pad_n,
            feat_pad_h,
            feat_pad_w,
            feat_pad_c,
            feat_pad_hh,
            feat_pad_ww,
        ] = prev_feat_pad.shape
        assert feat_pad_hh == kernel_h
        assert feat_pad_ww == kernel_w

        prev_feat_pad = prev_feat_pad.reshape(
            feat_pad_n * feat_pad_h * feat_pad_w, feat_pad_c, kernel_h,
            kernel_w)
        rand_index = list(range(prev_feat_pad.shape[0]))
        random.shuffle(rand_index)
        rand_index = rand_index[0:per_batch]
        X[per_batch * batch_idx:per_batch *
          (batch_idx + 1), :] = prev_feat_pad[rand_index, :]
        conv_feat_tmp = conv_feat.permute(0, 2, 3, 1).reshape(
            -1, conv_feat_c) - bias
        Y[per_batch * batch_idx:per_batch *
          (batch_idx + 1), :] = conv_feat_tmp[rand_index, :]

    handle_prev.remove()
    handle_conv.remove()

    ## ofwa init
    W_shape = W.shape
    X = X.cpu().numpy()
    Y = Y.cpu().numpy()
    W = W.reshape(W_shape[0], -1)
    if dw:
        B, alpha = BitSplitQuantizer(W.cpu().numpy(),
                                     bitwidth).ofwa_rr_dw(X, Y, num_epochs)
    else:
        B, alpha = BitSplitQuantizer(W.cpu().numpy(),
                                     bitwidth).ofwa_rr(X, Y, num_epochs)
    with open(prefix + '_rr_b30x400_e100.pkl', 'wb') as f:
        pickle.dump({'B': B, 'alpha': alpha}, f, pickle.HIGHEST_PROTOCOL)


def load_ofwa(conv, conv_quan, prefix=None):
    # for fc
    if not hasattr(conv, 'kernel_size'):
        W = conv.weight.data  # .cpu()
        W_shape = W.shape
        with open(prefix + '_fwa.pkl', 'rb') as f:
            B_alpha = pickle.load(f)
            B = B_alpha['B']
            alpha = B_alpha['alpha']
        W_r = np.multiply(B, np.expand_dims(alpha, 1)).reshape(W_shape)
        conv_quan.weight.data.copy_(torch.from_numpy(W_r))
        return

    # weights and bias
    W = conv.weight.data  # .cpu()
    W_shape = W.shape

    with open(prefix + '_rr_b30x400_e100.pkl', 'rb') as f:
        B_alpha = pickle.load(f)
        B = B_alpha['B']
        alpha = B_alpha['alpha']
    W_r = np.multiply(B, np.expand_dims(alpha, 1)).reshape(W_shape)
    conv_quan.weight.data.copy_(torch.from_numpy(W_r))


def save_state_dict(state_dict, path, filename='state_dict.pth'):
    saved_path = os.path.join(path, filename)
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if '.module.' in key:
            new_state_dict[key.replace('.module.',
                                       '.')] = state_dict[key].cpu()
        else:
            new_state_dict[key] = state_dict[key].cpu()
    torch.save(new_state_dict, saved_path)
