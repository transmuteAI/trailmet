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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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

from trailmet.algorithms.distill.distill import Distillation, ForwardHookManager

# Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer


class AttentionTransferLoss(nn.Module):
    """Class for loss used."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def attention_map(feature_map):
        """Compute the attention map from a feature map."""
        return F.normalize(feature_map.pow(2).mean(1).flatten(1))

    def compute_loss(self, teacher_feature_map, student_feature_map):
        """Compute the loss between teacher and student feature maps."""
        teacher_attention_map = self.attention_map(teacher_feature_map)
        student_attention_map = self.attention_map(student_feature_map)
        loss = (teacher_attention_map - student_attention_map).pow(2).mean()
        return loss

    def forward(self, feature_map_pairs):
        """feature_map_pairs: list of (teacher_feature_map, student_feature_map)"""
        loss = 0
        for teacher_feature_map, student_feature_map in feature_map_pairs:
            loss += self.compute_loss(teacher_feature_map, student_feature_map)
        return loss


class AttentionTransfer(Distillation):
    """Class to compress model using distillation via attention transfer.

    Parameters
    ----------
        teacher_model (object): Teacher model you want to use.
        student_model (object): Student model you want to use.
        dataloaders (dict): Dictionary with dataloaders for train, val and test. Keys: 'train', 'val', 'test'.
        kwargs (object): YAML safe loaded file with information like device, distill_args(teacher_layer_names, student_layer_names, etc).
    """

    def __init__(self, teacher_model, student_model, dataloaders, **kwargs):
        super(AttentionTransfer, self).__init__(**kwargs)
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.dataloaders = dataloaders
        self.kwargs = kwargs

        self.device = kwargs['DEVICE']
        self.beta = self.kwargs['DISTILL_ARGS'].get('BETA', 1000)

        # self.student_io_dict, self.teacher_io_dict = dict(), dict()
        self.teacher_layer_names = kwargs['DISTILL_ARGS'].get(
            'TEACHER_LAYER_NAMES')
        self.student_layer_names = kwargs['DISTILL_ARGS'].get(
            'STUDENT_LAYER_NAMES')
        self.forward_hook_manager_teacher = ForwardHookManager(self.device)
        self.forward_hook_manager_student = ForwardHookManager(self.device)

        self.ce_loss = nn.CrossEntropyLoss()
        self.at_loss = AttentionTransferLoss()

        self.epochs = kwargs['DISTILL_ARGS'].get('EPOCHS', 200)
        self.lr = kwargs['DISTILL_ARGS'].get('LR', 0.1)

        self.wandb_monitor = self.kwargs.get('wandb', 'False')
        self.dataset_name = dataloaders['train'].dataset.__class__.__name__
        self.save = './checkpoints/'

        self.name = '_'.join([
            self.dataset_name,
            f'{self.epochs}',
            f'{self.lr}',
            datetime.now().strftime('%b-%d_%H:%M:%S'),
        ])

        os.makedirs(f'{os.getcwd()}/logs/Attention_Transfer', exist_ok=True)
        os.makedirs(self.save, exist_ok=True)
        self.logger_file = f'{os.getcwd()}/logs/Attention_Transfer/{self.name}.log'

        logging.basicConfig(
            filename=self.logger_file,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )

        logger.info(f'Experiment Arguments: {self.kwargs}')

        if self.wandb_monitor:
            wandb.init(project='Trailmet Attention_Transfer', name=self.name)
            wandb.config.update(self.kwargs)

    def compress_model(self):
        """Function to transfer knowledge from teacher to student."""
        # include teacher training options
        self.distill(
            self.teacher_model,
            self.student_model,
            self.dataloaders,
            **self.kwargs['DISTILL_ARGS'],
        )

    def distill(self, teacher_model, student_model, dataloaders, **kwargs):
        print('=====> TRAINING STUDENT NETWORK <=====')
        logger.info('=====> TRAINING STUDENT NETWORK <=====')

        self.register_hooks()

        test_only = kwargs.get('TEST_ONLY', False)
        weight_decay = kwargs.get('WEIGHT_DECAY', 0.0005)
        milestones = kwargs.get('MILESTONES', [82, 123])
        gamma = kwargs.get('GAMMA', 0.1)

        optimizer = torch.optim.SGD(
            student_model.parameters(),
            lr=self.lr,
            weight_decay=weight_decay,
            momentum=0.9,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=milestones,
                                                         gamma=gamma,
                                                         verbose=False)

        criterion = self.criterion

        best_top1_acc = 0

        if test_only == False:
            epochs_list = []
            val_top1_acc_list = []
            val_top5_acc_list = []
            for epoch in range(self.epochs):
                t_loss = self.train_one_epoch(
                    teacher_model,
                    student_model,
                    dataloaders['train'],
                    criterion,
                    optimizer,
                    epoch,
                )

                valid_loss, valid_top1_acc, valid_top5_acc = self.test(
                    teacher_model, student_model, dataloaders['val'],
                    criterion, epoch)

                # use conditions for different schedulers e.g. ReduceLROnPlateau needs scheduler.step(v_loss)
                scheduler.step()

                is_best = False
                if valid_top1_acc > best_top1_acc:
                    best_top1_acc = valid_top1_acc
                    is_best = True

                save_checkpoint(
                    {
                        'epoch': epoch,
                        'state_dict': student_model.state_dict(),
                        'best_top1_acc': best_top1_acc,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    },
                    is_best,
                    self.save,
                )

                if self.wandb_monitor:
                    wandb.log({'best_top1_acc': best_top1_acc})

                epochs_list.append(epoch)
                val_top1_acc_list.append(valid_top1_acc.cpu().numpy())
                val_top5_acc_list.append(valid_top5_acc.cpu().numpy())

                df_data = np.array([
                    epochs_list,
                    val_top1_acc_list,
                    val_top5_acc_list,
                ]).T
                df = pd.DataFrame(
                    df_data,
                    columns=[
                        'Epochs',
                        'Validation Top1',
                        'Validation Top5',
                    ],
                )
                df.to_csv(
                    f'{os.getcwd()}/logs/Attention_Transfer/{self.name}.csv',
                    index=False,
                )

    def train_one_epoch(self, teacher_model, student_model, dataloader,
                        loss_fn, optimizer, epoch):
        teacher_model.eval()
        student_model.train()

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        end = time.time()

        epoch_iterator = tqdm(
            dataloader,
            desc=
            'Training student network Epoch [X] (X / X Steps) (batch time=X.Xs) (data time=X.Xs) (loss=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        for i, (images, labels) in enumerate(epoch_iterator):
            data_time.update(time.time() - end)
            images = images.to(self.device, dtype=torch.float)
            labels = labels.to(self.device)

            teacher_preds = teacher_model(images)
            student_preds = student_model(images)

            teacher_io_dict = self.forward_hook_manager_teacher.pop_io_dict()
            student_io_dict = self.forward_hook_manager_student.pop_io_dict()
            feature_map_pairs = []
            for teacher_layer, student_layer in zip(self.teacher_layer_names,
                                                    self.student_layer_names):
                feature_map_pairs.append((
                    teacher_io_dict[teacher_layer]['output'],
                    student_io_dict[student_layer]['output'],
                ))

            loss = loss_fn(teacher_preds, student_preds, feature_map_pairs,
                           labels)
            n = images.size(0)
            losses.update(loss.item(), n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            epoch_iterator.set_description(
                'Training student network Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f)'
                % (
                    epoch,
                    (i + 1),
                    len(dataloader),
                    batch_time.val,
                    data_time.val,
                    losses.val,
                ))

            logger.info(
                'Training student network Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f)'
                % (
                    epoch,
                    (i + 1),
                    len(dataloader),
                    batch_time.val,
                    data_time.val,
                    losses.val,
                ))

            if self.wandb_monitor:
                wandb.log({
                    'train_loss': losses.val,
                })

        return losses.avg

    def test(self, teacher_model, student_model, dataloader, loss_fn, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        epoch_iterator = tqdm(
            dataloader,
            desc=
            'Validating student network Epoch [X] (X / X Steps) (batch time=X.Xs) (loss=X.X) (top1=X.X) (top5=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        teacher_model.eval()
        student_model.eval()

        with torch.no_grad():
            end = time.time()

            for i, (images, labels) in enumerate(epoch_iterator):
                images = images.to(self.device, dtype=torch.float)
                labels = labels.to(self.device)

                teacher_preds = teacher_model(images)
                student_preds = student_model(images)

                teacher_io_dict = self.forward_hook_manager_teacher.pop_io_dict(
                )
                student_io_dict = self.forward_hook_manager_student.pop_io_dict(
                )
                feature_map_pairs = []
                for teacher_layer, student_layer in zip(
                        self.teacher_layer_names, self.student_layer_names):
                    feature_map_pairs.append((
                        teacher_io_dict[teacher_layer]['output'],
                        student_io_dict[student_layer]['output'],
                    ))

                loss = loss_fn(teacher_preds, student_preds, feature_map_pairs,
                               labels)

                pred1, pred5 = accuracy(student_preds, labels, topk=(1, 5))

                n = images.size(0)
                losses.update(loss.item(), n)
                top1.update(pred1[0], n)
                top5.update(pred5[0], n)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                epoch_iterator.set_description(
                    'Validating student network Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
                    % (
                        epoch,
                        (i + 1),
                        len(dataloader),
                        batch_time.val,
                        losses.val,
                        top1.val,
                        top5.val,
                    ))

                logger.info(
                    'Validating student network Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
                    % (
                        epoch,
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

    def criterion(self, teacher_preds, student_preds, feature_map_pairs,
                  labels):
        ce_loss = self.ce_loss(student_preds, labels)
        at_loss = self.at_loss(feature_map_pairs)
        return ce_loss + self.beta * at_loss

    def register_hooks(self):
        for layer in self.teacher_layer_names:
            self.forward_hook_manager_teacher.add_hook(self.teacher_model,
                                                       layer,
                                                       requires_input=False,
                                                       requires_output=True)

        for layer in self.student_layer_names:
            self.forward_hook_manager_student.add_hook(self.student_model,
                                                       layer,
                                                       requires_input=False,
                                                       requires_output=True)
