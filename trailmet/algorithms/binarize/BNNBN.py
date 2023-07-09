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
import os
import sys
import numpy as np
import time, datetime
import torch
import logging
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

import timm
from .utils import DistributionLoss
from trailmet.utils import (
    AverageMeter,
    save_checkpoint,
    accuracy,
    CrossEntropyLabelSmooth,
)
import torchvision.models as models
from timm.utils.agc import adaptive_clip_grad
import logging
from datetime import datetime
from tqdm import tqdm
import wandb
import pandas as pd

logger = logging.getLogger(__name__)


class BNNBN:
    """
    References
    ----------

    Parameters
    ----------
        model (object): A pytorch model you want to use.
        dataloaders (dict): Dictionary with dataloaders for train, val and test. Keys: 'train', 'val', 'test'.
        kwargs (object): YAML safe loaded file with information like batch_size, optimizer, epochs, momentum, etc.
    """

    def __init__(self, model, dataloaders, **kwargs):
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.test_loader = dataloaders['test']
        self.model_student = model
        self.kwargs = kwargs
        self.dataset = self.kwargs.get('DATASET', 'c100')
        self.num_classes = self.kwargs.get('num_classes', 100)
        self.loss_type = self.kwargs.get('loss_type', 'kd')
        self.teacher = self.kwargs.get('teacher', 'resnet34')
        self.teacher_weight = self.kwargs.get('teacher_weight', '')
        self.label_smooth = self.kwargs.get('label_smooth', '0.1')
        self.pretrained = self.kwargs.get('pretrained', '')
        self.resume = self.kwargs.get('resume', 'False')
        self.save = self.kwargs.get('save', '')
        self.epochs = self.kwargs.get('epoch', '120')
        self.agc = self.kwargs.get('agc', 'True')
        self.clip_value = self.kwargs.get('clip_value', '0.04')
        self.weight_decay = self.kwargs.get('weight_decay', '0')
        self.learning_rate = self.kwargs.get('learning_rate', '0.001')

        self.wandb_monitor = self.kwargs.get('wandb', 'False')
        self.dataset_name = dataloaders['train'].dataset.__class__.__name__

        self.name = '_'.join([
            self.dataset_name,
            f'{self.epochs}',
            f'{self.learning_rate}',
            datetime.now().strftime('%b-%d_%H:%M:%S'),
        ])

        os.makedirs(f'{os.getcwd()}/logs/BNNBN', exist_ok=True)
        self.logger_file = f'{os.getcwd()}/logs/BNNBN/{self.name}.log'

        logging.basicConfig(
            filename=self.logger_file,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )

        logger.info(f'Experiment Arguments: {self.kwargs}')

        if self.wandb_monitor:
            wandb.init(project='Trailmet BNNBN', name=self.name)
            wandb.config.update(self.kwargs)

    def compress_model(self):
        """Function for compressing the model."""
        if not torch.cuda.is_available():
            sys.exit(1)
        start_t = time.time()

        cudnn.benchmark = True
        cudnn.enabled = True

        self.model_student.cuda()
        print(f'Student Model:{self.model_student}')

        # load teacher model
        if self.loss_type == 'kd':
            print('Loading teacher model')
            logger.info('Loading teacher model')
            if not 'nfnet' in self.teacher:
                model_teacher = models.__dict__[self.teacher](pretrained=True)
                classes_in_teacher = model_teacher.fc.out_features
                num_features = model_teacher.fc.in_features
            else:
                model_teacher = timm.create_model(self.teacher,
                                                  pretrained=True)
                classes_in_teacher = model_teacher.head.fc.out_features
                num_features = model_teacher.head.fc.in_features

            if not classes_in_teacher == self.num_classes:
                print('* change fc layers in teacher')
                if not 'nfnet' in self.teacher:
                    model_teacher.fc = nn.Linear(num_features,
                                                 self.num_classes)
                else:
                    model_teacher.head.fc = nn.Linear(num_features,
                                                      self.num_classes)
                print('* loading pretrained teacher weight from {}'.format(
                    self.teacher_weight))
                pretrain_teacher = torch.load(self.teacher_weight,
                                              map_location='cpu')['state_dict']
                model_teacher.load_state_dict(pretrain_teacher)

            print('Teacher Model: {}'.format(model_teacher))
            logger.info('Teacher Model: {}'.format(model_teacher))
            model_teacher.cuda()
            for p in model_teacher.parameters():
                p.requires_grad = False
            model_teacher.eval()

        # criterion
        criterion = nn.CrossEntropyLoss().cuda()
        criterion_smooth = CrossEntropyLabelSmooth(self.num_classes,
                                                   self.label_smooth).cuda()
        criterion_kd = DistributionLoss()

        # optimizer
        all_parameters = self.model_student.parameters()
        weight_parameters = []
        for pname, p in self.model_student.named_parameters():
            if p.ndimension() == 4 or 'conv' in pname:
                weight_parameters.append(p)
        weight_parameters_id = list(map(id, weight_parameters))
        other_parameters = list(
            filter(lambda p: id(p) not in weight_parameters_id,
                   all_parameters))

        optimizer = torch.optim.Adam(
            [
                {
                    'params': other_parameters
                },
                {
                    'params': weight_parameters,
                    'weight_decay': self.weight_decay
                },
            ],
            lr=self.learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: (1.0 - step / self.epochs), last_epoch=-1)
        start_epoch = 0
        best_top1_acc = 0

        if self.pretrained:
            print('Loading pretrained weight {}'.format(self.pretrained))
            logger.info('Loading pretrained weight {}'.format(self.pretrained))
            pretrain_student = torch.load(self.pretrained)
            if 'state_dict' in pretrain_student.keys():
                pretrain_student = pretrain_student['state_dict']

            for key in pretrain_student.keys():
                if not key in self.model_student.state_dict().keys():
                    print('unload key: {}'.format(key))

            self.model_student.load_state_dict(pretrain_student, strict=False)

        if self.resume:
            checkpoint_tar = os.path.join(self.save, 'checkpoint.pth.tar')
            if os.path.exists(checkpoint_tar):
                print('Loading from the checkpoint {}'.format(checkpoint_tar))
                logger.info(
                    'Loading from the checkpoint {}'.format(checkpoint_tar))
                checkpoint = torch.load(checkpoint_tar)
                start_epoch = checkpoint['epoch']
                best_top1_acc = checkpoint['best_top1_acc']
                self.model_student.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                print('Loaded checkpoint {} epoch = {}'.format(
                    checkpoint_tar, checkpoint['epoch']))
                logger.info('Loaded checkpoint {} epoch = {}'.format(
                    checkpoint_tar, ['epoch']))
            else:
                raise ValueError('no checkpoint for resume')

        if self.loss_type == 'kd':
            if not classes_in_teacher == self.num_classes:
                self.validate('teacher', self.val_loader, model_teacher,
                              criterion)

        # train the model
        epoch = start_epoch
        epochs_list = []
        train_top1_acc_list = []
        train_top5_acc_list = []
        val_top1_acc_list = []
        val_top5_acc_list = []

        while epoch < self.epochs:
            if self.loss_type == 'kd':
                train_obj, train_top1_acc, train_top5_acc = self.train_kd(
                    epoch,
                    self.train_loader,
                    self.model_student,
                    model_teacher,
                    criterion_kd,
                    optimizer,
                    scheduler,
                )

                train_top1_acc_list.append(train_top1_acc)
                train_top5_acc_list.append(train_top5_acc.cpu().numpy())
            elif self.loss_type == 'ce':
                train_obj, train_top1_acc, train_top5_acc = self.train(
                    epoch,
                    self.train_loader,
                    self.model_student,
                    criterion,
                    optimizer,
                    scheduler,
                )

                train_top1_acc_list.append(train_top1_acc)
                train_top5_acc_list.append(train_top5_acc)
            elif self.loss_type == 'ls':
                train_obj, train_top1_acc, train_top5_acc = self.train(
                    epoch,
                    self.train_loader,
                    self.model_student,
                    criterion_smooth,
                    optimizer,
                    scheduler,
                )

                train_top1_acc_list.append(train_top1_acc)
                train_top5_acc_list.append(train_top5_acc)
            else:
                raise ValueError('unsupport loss_type')

            valid_obj, valid_top1_acc, valid_top5_acc = self.validate(
                epoch, self.val_loader, self.model_student, criterion)

            epochs_list.append(epoch)
            val_top1_acc_list.append(valid_top1_acc.cpu().numpy())
            val_top5_acc_list.append(valid_top5_acc.cpu().numpy())

            is_best = False
            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                is_best = True

            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': self.model_student.state_dict(),
                    'best_top1_acc': best_top1_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                },
                is_best,
                self.save,
            )

            epoch += 1

            if self.wandb_monitor:
                wandb.log({'best_top1_acc': best_top1_acc})

        training_time = (time.time() - start_t) / 3600
        print('Total training time = {} hours'.format(training_time))
        print('Best acc = {}'.format(best_top1_acc))
        logger.info('Total training time = {} hours'.format(training_time))
        logger.info('Best acc = {}'.format(best_top1_acc))

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
        df.to_csv(f'{os.getcwd()}/logs/BNNBN/{self.name}.csv', index=False)

    def train_kd(
        self,
        epoch,
        train_loader,
        model_student,
        model_teacher,
        criterion,
        optimizer,
        scheduler,
    ):
        """Function for training KD."""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        model_student.train()
        model_teacher.eval()
        end = time.time()
        scheduler.step()

        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('learning_rate:', cur_lr)

        epoch_iterator = tqdm(
            train_loader,
            desc=
            'Training KD in BNNBN Epoch [X] (X / X Steps) (batch time=X.Xs) (data time=X.Xs) (loss=X.X) (top1=X.X) (top5=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        for i, (images, target) in enumerate(epoch_iterator):
            data_time.update(time.time() - end)
            images = images.cuda()
            target = target.cuda()

            # compute outputy
            logits_student = model_student(images)
            logits_teacher = model_teacher(images)
            loss = criterion(logits_student, logits_teacher)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)  # accumulated loss
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            # clip gradient if necessary
            if self.agc:
                parameters_list = []
                for name, p in model_student.named_parameters():
                    if not 'fc' in name:
                        parameters_list.append(p)
                adaptive_clip_grad(parameters_list,
                                   clip_factor=self.clip_value)

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            epoch_iterator.set_description(
                'Training KD in BNNBN Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
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
                'Training KD in BNNBN Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
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
                    'train_loss_KD': losses.val,
                    'train_top1_acc_KD': top1.val,
                    'train_top5_acc_KD': top5.val,
                })

        return losses.avg, top1.avg, top5.avg

    def train(self, epoch, train_loader, model_student, criterion, optimizer,
              scheduler):
        """Training Function."""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        epoch_iterator = tqdm(
            train_loader,
            desc=
            'Training in BNNBN Epoch [X] (X / X Steps) (batch time=X.Xs) (data time=X.Xs) (loss=X.X) (top1=X.X) (top5=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        model_student.train()
        end = time.time()
        scheduler.step()

        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']

        for i, (images, target) in enumerate(epoch_iterator):
            data_time.update(time.time() - end)
            images = images.cuda()
            target = target.cuda()

            # compute outputy
            logits_student = model_student(images)
            loss = criterion(logits_student, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)  # accumulated loss
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            # clip gradient if necessary
            if self.agc:
                parameters_list = []
                for name, p in model_student.named_parameters():
                    if not 'fc' in name:
                        parameters_list.append(p)
                adaptive_clip_grad(parameters_list,
                                   clip_factor=self.clip_value)

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            epoch_iterator.set_description(
                'Training in BNNBN Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
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
                'Training in BNNBN Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
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

    def validate(self, epoch, val_loader, model, criterion):
        """Validate Function."""
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        epoch_iterator = tqdm(
            val_loader,
            desc=
            'Validating in BNNBN Epoch [X] (X / X Steps) (batch time=X.Xs) (loss=X.X) (top1=X.X) (top5=X.X)',
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
