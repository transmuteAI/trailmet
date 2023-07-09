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
from trailmet.algorithms.binarize.binarize import BaseBinarize
from trailmet.models.resnet import ResNetCifar
from trailmet.models.resnet import ResNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import DistributionLoss
from trailmet.utils import AverageMeter, save_checkpoint, accuracy

import os
import logging
from datetime import datetime
from tqdm import tqdm
import wandb
import pandas as pd
import time
import numpy as np

logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BinaryActivation(nn.Module):
    """Class for BinaryActivation."""

    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        # out_e1 = (x^2 + 2*x)
        # out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(
            torch.float32) + (x * x + 2 * x) * (1 - mask1.type(torch.float32))
        out2 = out1 * mask2.type(
            torch.float32) + (-x * x + 2 * x) * (1 - mask2.type(torch.float32))
        out3 = out2 * mask3.type(
            torch.float32) + 1 * (1 - mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class LearnableBias(nn.Module):
    """Class for LearnableBias."""

    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1),
                                 requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class HardBinaryConv(nn.Module):

    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        # self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001,
                                   requires_grad=True)

    def forward(self, x):
        # real_weights = self.weights.view(self.shape)
        real_weights = self.weight
        scaling_factor = torch.mean(
            torch.mean(torch.mean(abs(real_weights), dim=3, keepdim=True),
                       dim=2,
                       keepdim=True),
            dim=1,
            keepdim=True,
        )
        # print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = (binary_weights_no_grad.detach() -
                          cliped_weights.detach() + cliped_weights)
        # print(binary_weights, flush=True)
        y = F.conv2d(x,
                     binary_weights,
                     stride=self.stride,
                     padding=self.padding)

        return y


class BasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1, self).__init__()

        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out


class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock2, self).__init__()

        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out


class ReActNet(BaseBinarize):
    """
    References
    ----------

    Parameters
    ----------
        teacher (object): Teacher model you want to use.
        model (object): A pytorch model you want to use.
        dataloaders (dict): Dictionary with dataloaders for train, val and test. Keys: 'train', 'val', 'test'.
        kwargs (object): YAML safe loaded file with information like batch_size, optimizer, epochs, momentum, etc.
    """

    def __init__(self, teacher, model, dataloaders, **kwargs):
        super(ReActNet, self).__init__(**kwargs)
        self.teacher = teacher
        self.model = model
        self.layers = model.layers_size
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataloaders = dataloaders
        self.num_classes = model.num_classes
        self.insize = model.insize
        self.kwargs = kwargs

        self.batch_size1 = self.kwargs['ReActNet1_ARGS'].get('batch_size', 128)
        self.epochs1 = self.kwargs['ReActNet1_ARGS'].get('epochs', 128)
        self.learning_rate1 = self.kwargs['ReActNet1_ARGS'].get(
            'learning_rate', 2.5e-3)
        self.momentum1 = self.kwargs['ReActNet1_ARGS'].get('momentum', 0.9)
        self.weight_decay1 = self.kwargs['ReActNet1_ARGS'].get(
            'weight_decay', 1e-5)
        self.label_smooth1 = self.kwargs['ReActNet1_ARGS'].get(
            'label_smooth', 0.1)

        self.batch_size2 = self.kwargs['ReActNet2_ARGS'].get('batch_size', 128)
        self.epochs2 = self.kwargs['ReActNet2_ARGS'].get('epochs', 128)
        self.learning_rate2 = self.kwargs['ReActNet2_ARGS'].get(
            'learning_rate', 2.5e-3)
        self.momentum2 = self.kwargs['ReActNet2_ARGS'].get('momentum', 0.9)
        self.weight_decay2 = self.kwargs['ReActNet2_ARGS'].get(
            'weight_decay', 0)
        self.label_smooth2 = self.kwargs['ReActNet2_ARGS'].get(
            'label_smooth', 0.1)

        self.wandb_monitor = self.kwargs.get('wandb', 'False')
        self.dataset_name = dataloaders['train'].dataset.__class__.__name__

        self.name = '_'.join([
            self.dataset_name,
            f'{self.epochs1}',
            f'{self.learning_rate1}',
            datetime.now().strftime('%b-%d_%H:%M:%S'),
        ])

        os.makedirs(f'{os.getcwd()}/logs/ReActNet', exist_ok=True)
        self.logger_file = f'{os.getcwd()}/logs/ReActNet/{self.name}.log'

        logging.basicConfig(
            filename=self.logger_file,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )

        logger.info(f'Experiment Arguments: {self.kwargs}')

        if self.wandb_monitor:
            wandb.init(project='Trailmet ReActNet', name=self.name)
            wandb.config.update(self.kwargs)

    def make_model1(self):
        self.layers = [i * 2 for i in self.layers]

        if len(self.layers) == 3:
            new_model = ResNetCifar(
                BasicBlock1,
                self.layers,
                width=1,
                num_classes=self.num_classes,
                insize=self.insize,
            )
        else:
            new_model = ResNet(
                BasicBlock1,
                self.layers,
                width=1,
                num_classes=self.num_classes,
                insize=self.insize,
            )

        return new_model

    def make_model2(self):
        self.layers = [i * 2 for i in self.layers]

        if len(self.layers) == 3:
            new_model = ResNetCifar(
                BasicBlock2,
                self.layers,
                width=1,
                num_classes=self.num_classes,
                insize=self.insize,
            )
        else:
            new_model = ResNet(
                BasicBlock2,
                self.layers,
                width=1,
                num_classes=self.num_classes,
                insize=self.insize,
            )

        return new_model

    def train_one_epoch(self, model, teacher, scheduler, criterion, optimizer,
                        epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        end = time.time()
        scheduler.step()

        train_loader = self.dataloaders['train']

        epoch_iterator = tqdm(
            train_loader,
            desc=
            'Training Epoch [X] (X / X Steps) (batch time=X.Xs) (data time=X.Xs) (loss=X.X) (top1=X.X) (top5=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        for i, (images, target) in enumerate(epoch_iterator):
            data_time.update(time.time() - end)
            images = images.cuda()
            target = target.cuda()

            # compute outputy
            logits_student = model(images)
            logits_teacher = teacher(images)
            loss = criterion(logits_student, logits_teacher)

            prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)  # accumulated loss
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            epoch_iterator.set_description(
                'Training Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
                % (
                    epoch,
                    (i + 1),
                    len(train_loader),
                    batch_time.val,
                    data_time.val,
                    losses.val,
                    top1.val,
                    top5.val,
                ))

            logger.info(
                'Training Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
                % (
                    epoch,
                    (i + 1),
                    len(train_loader),
                    batch_time.val,
                    data_time.val,
                    losses.val,
                    top1.val,
                    top5.val,
                ))

            if self.wandb_monitor:
                wandb.log({
                    'train_loss': losses.val,
                    'train_top1_acc': top1.val,
                    'train_top5_acc': top5.val,
                })

        return losses.avg, top1.avg, top5.avg

    def base_train(self, model, teacher, epochs, criterion, scheduler,
                   optimizer, step):
        model.train()
        teacher = teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        epoch = 0
        epochs_list = []
        train_top1_acc_list = []
        train_top5_acc_list = []
        val_top1_acc_list = []
        val_top5_acc_list = []
        while epoch < epochs:
            train_obj, train_top1_acc, train_top5_acc = self.train_one_epoch(
                model, teacher, scheduler, criterion, optimizer, epoch)

            train_top1_acc_list.append(train_top1_acc)
            train_top5_acc_list.append(train_top5_acc)

            valid_loss, valid_top1_acc, valid_top5_acc = self.validate(
                epoch, self.dataloaders['val'], model, criterion)

            epochs_list.append(epoch)
            val_top1_acc_list.append(valid_top1_acc.cpu().numpy())
            val_top5_acc_list.append(valid_top5_acc.cpu().numpy())

            print.info(
                'Average Train Loss-{0}\tTop-1 Train Accuracy-{1}\tTop-5 Train Accuracy-{2}\tTop1 Validation Accuracy-{3}\tTop5 Validation Accuracy-{4}\tValidation Loss-{5}'
                .format(
                    train_obj,
                    train_top1_acc,
                    train_top5_acc,
                    valid_top1_acc,
                    valid_top5_acc,
                    valid_loss,
                ))

            logger.info(
                'Average Train Loss-{0}\tTop-1 Train Accuracy-{1}\tTop-5 Train Accuracy-{2}\tTop1 Validation Accuracy-{3}\tTop5 Validation Accuracy-{4}\tValidation Loss-{5}'
                .format(
                    train_obj,
                    train_top1_acc,
                    train_top5_acc,
                    valid_top1_acc,
                    valid_top5_acc,
                    valid_loss,
                ))
            epoch += 1

        df_data = np.array([
            epochs_list,
            train_top1_acc_list,
            train_top5_acc_list,
            val_top1_acc_list,
            val_top5_acc_list,
        ]).T
        df = pd.DataFrame(
            df_data,
            columns=[
                'Epochs',
                'Train Top1',
                'Train Top5',
                'Validation Top1',
                'Validation Top5',
            ],
        )
        df.to_csv(f'{os.getcwd()}/logs/ReActNet/Step-{step}-{self.name}.csv',
                  index=False)

        return model

    def train1(self):
        print('Step-1 Training with activations binarized for {} epochs\n'.
              format(self.epochs1))
        logger.info(
            'Step-1 Training with activations binarized for {} epochs\n'.
            format(self.epochs1))
        model = self.make_model1()
        model = model.to(self.device)
        teacher = self.teacher.to(self.device)

        all_parameters = model.parameters()
        weight_parameters = []
        for pname, p in model.named_parameters():
            if p.ndimension() == 4 or 'conv' in pname:
                weight_parameters.append(p)
        weight_parameters_id = list(map(id, weight_parameters))
        other_parameters = list(
            filter(lambda p: id(p) not in weight_parameters_id,
                   all_parameters))

        criterion = DistributionLoss()
        optimizer = torch.optim.Adam(
            [
                {
                    'params': other_parameters
                },
                {
                    'params': weight_parameters,
                    'weight_decay': self.weight_decay1
                },
            ],
            lr=self.learning_rate1,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: (1.0 - step / self.epochs1), last_epoch=-1)

        model = self.base_train(model, teacher, self.epochs1, criterion,
                                scheduler, optimizer, '1')
        return model

    def train2(self, model):
        print(
            'Step-2 Training with both activations and weights binarized for {} epochs'
            .format(self.epochs2))
        logger.info(
            'Step-2 Training with both activations and weights binarized for {} epochs'
            .format(self.epochs2))
        # model = model.to(self.device)
        teacher = self.teacher.to(self.device)

        all_parameters = model.parameters()
        weight_parameters = []
        for pname, p in model.named_parameters():
            if p.ndimension() == 4 or 'conv' in pname:
                weight_parameters.append(p)
        weight_parameters_id = list(map(id, weight_parameters))
        other_parameters = list(
            filter(lambda p: id(p) not in weight_parameters_id,
                   all_parameters))

        criterion = DistributionLoss()
        optimizer = torch.optim.Adam(
            [
                {
                    'params': other_parameters
                },
                {
                    'params': weight_parameters,
                    'weight_decay': self.weight_decay2
                },
            ],
            lr=self.learning_rate2,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: (1.0 - step / self.epochs2), last_epoch=-1)

        model = self.base_train(model, teacher, self.epochs2, criterion,
                                scheduler, optimizer, '2')
        return model

    def validate(self, epoch, val_loader, model, criterion):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        epoch_iterator = tqdm(
            val_loader,
            desc=
            'Validating Epoch [X] (X / X Steps) (batch time=X.Xs) (loss=X.X) (top1=X.X) (top5=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        # switch to evaluation mode
        model.eval()
        with torch.no_grad():
            end = time.time()

            for i, (images, target) in enumerate(epoch_iterator):
                images = images.cuda()
                target = target.cuda()

                # compute output
                logits = model(images)
                loss = criterion(logits, target)

                # measure accuracy and record loss
                pred1, pred5 = accuracy(logits, target, topk=(1, 5))
                n = images.size(0)
                losses.update(loss.item(), n)
                top1.update(pred1[0], n)
                top5.update(pred5[0], n)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                epoch_iterator.set_description(
                    'Validating Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
                    % (
                        epoch,
                        (i + 1),
                        len(val_loader),
                        batch_time.val,
                        losses.val,
                        top1.val,
                        top5.val,
                    ))

                logger.info(
                    'Validating Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
                    % (
                        epoch,
                        (i + 1),
                        len(val_loader),
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

    def compress_model(self):
        model1 = self.train1()
        model2 = self.make_model2()
        model2 = model2.to(self.device)
        model2.load_state_dict(model1.state_dict(), strict=False)

        fin_output = self.train2(model2)

        return fin_output
