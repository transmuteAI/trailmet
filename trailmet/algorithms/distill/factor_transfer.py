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
from trailmet.algorithms.distill.distill import Distillation, ForwardHookManager

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

# Paraphrasing Complex Network: Network Compression via Factor Transfer


class FactorTransferLoss(nn.Module):

    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.criterion = nn.L1Loss()
        self.criterion_ce = nn.CrossEntropyLoss()

    @staticmethod
    def FT(x):
        return F.normalize(x.view(x.size(0), -1))

    def forward(self, factor_teacher, factor_student, logits, labels):
        loss = self.criterion_ce(logits, labels)
        loss += self.beta * self.criterion(self.FT(factor_student),
                                           self.FT(factor_teacher.detach()))
        return loss


class Paraphraser(nn.Module):
    """Paraphraser Class."""

    def __init__(self, in_planes, planes, stride=1):
        super(Paraphraser, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.1)
        #       self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv0 = nn.Conv2d(in_planes,
                               in_planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        #       self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        #       self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        #       self.bn0_de = nn.BatchNorm2d(planes)
        self.deconv0 = nn.ConvTranspose2d(planes,
                                          planes,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=True)
        #       self.bn1_de = nn.BatchNorm2d(in_planes)
        self.deconv1 = nn.ConvTranspose2d(planes,
                                          in_planes,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=True)
        #       self.bn2_de = nn.BatchNorm2d(in_planes)
        self.deconv2 = nn.ConvTranspose2d(in_planes,
                                          in_planes,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=True)

    #### Mode 0 - throw encoder and decoder (reconstruction)
    #### Mode 1 - extracting teacher factors
    def forward(self, x, mode):
        if mode == 0:
            ## encoder
            out = self.leakyrelu((self.conv0(x)))
            out = self.leakyrelu((self.conv1(out)))
            out = self.leakyrelu((self.conv2(out)))
            ## decoder
            out = self.leakyrelu((self.deconv0(out)))
            out = self.leakyrelu((self.deconv1(out)))
            out = self.leakyrelu((self.deconv2(out)))

        if mode == 1:
            out = self.leakyrelu((self.conv0(x)))
            out = self.leakyrelu((self.conv1(out)))
            out = self.leakyrelu((self.conv2(out)))

        ## only throw decoder
        if mode == 2:
            out = self.leakyrelu((self.deconv0(x)))
            out = self.leakyrelu((self.deconv1(out)))
            out = self.leakyrelu((self.deconv2(out)))
        return out


class Translator(nn.Module):
    """Translator Class."""

    def __init__(self, in_planes, planes, stride=1):
        super(Translator, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.1)
        #       self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv0 = nn.Conv2d(in_planes,
                               in_planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        #       self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        #       self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)

    def forward(self, x):
        out = self.leakyrelu((self.conv0(x)))
        out = self.leakyrelu((self.conv1(out)))
        out = self.leakyrelu((self.conv2(out)))
        return out


class FactorTransfer(Distillation):
    """Class to compress model using distillation via factor transfer.

    Parameters
    ----------
        teacher_model (object): Teacher model you want to use.
        student_model (object): Student model you want to use.
        dataloaders (dict): Dictionary with dataloaders for train, val and test. Keys: 'train', 'val', 'test'.
        paraphraser (object): Paraphrase model
        kwargs (object): YAML safe loaded file with information like distill_args(teacher_layer_names, student_layer_names, etc).
    """

    def __init__(self,
                 teacher_model,
                 student_model,
                 dataloaders,
                 paraphraser=None,
                 **kwargs):
        super(FactorTransfer, self).__init__(**kwargs)
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.paraphraser = paraphraser
        self.dataloaders = dataloaders
        self.kwargs = kwargs
        self.beta = self.kwargs['DISTILL_ARGS'].get('BETA', 500)
        self.verbose = self.kwargs['DISTILL_ARGS'].get('VERBOSE', False)

        # self.student_io_dict, self.teacher_io_dict = dict(), dict()
        self.teacher_layer_name = kwargs['DISTILL_ARGS'].get(
            'TEACHER_LAYER_NAME')
        self.student_layer_name = kwargs['DISTILL_ARGS'].get(
            'STUDENT_LAYER_NAME')
        self.forward_hook_manager_teacher = ForwardHookManager(self.device)
        self.forward_hook_manager_student = ForwardHookManager(self.device)

        self.ft_loss = FactorTransferLoss(self.beta)
        self.l1_loss = nn.L1Loss()

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

        os.makedirs(f'{os.getcwd()}/logs/Factor_Transfer', exist_ok=True)
        os.makedirs(self.save, exist_ok=True)
        self.logger_file = f'{os.getcwd()}/logs/Factor_Transfer/{self.name}.log'

        logging.basicConfig(
            filename=self.logger_file,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )

        logger.info(f'Experiment Arguments: {self.kwargs}')

        if self.wandb_monitor:
            wandb.init(project='Trailmet Factor_Transfer', name=self.name)
            wandb.config.update(self.kwargs)

    def compress_model(self):
        """Function to transfer knowledge from teacher to student."""

        # include teacher training options

        self.register_hooks()

        if self.paraphraser == None:
            if 'paraphraser' in self.dataloaders:
                self.train_paraphraser(
                    self.teacher_model,
                    self.dataloaders['paraphraser'],
                    **self.kwargs['PARAPHRASER'],
                )
            else:
                self.train_paraphraser(self.teacher_model, self.dataloaders,
                                       **self.kwargs['PARAPHRASER'])

        self.distill(
            self.teacher_model,
            self.student_model,
            self.paraphraser,
            self.dataloaders,
            **self.kwargs['DISTILL_ARGS'],
        )

    def distill(self, teacher_model, student_model, paraphraser, dataloaders,
                **kwargs):
        print('=====> TRAINING STUDENT NETWORK <=====')
        logger.info('=====> TRAINING STUDENT NETWORK <=====')

        self.register_hooks()
        test_only = kwargs.get('TEST_ONLY', False)
        weight_decay = kwargs.get('WEIGHT_DECAY', 0.0005)
        milestones = kwargs.get('MILESTONES', [82, 123])
        gamma = kwargs.get('GAMMA', 0.1)

        in_planes = kwargs.get('IN_PLANES', 64)
        rate = kwargs.get('RATE', 0.5)
        planes = kwargs.get('planes', int(in_planes * rate))

        translator = Translator(in_planes, planes)
        translator.to(self.device)

        # dont hard code this
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
        optimizer_translator = torch.optim.SGD(translator.parameters(),
                                               lr=self.lr,
                                               weight_decay=weight_decay,
                                               momentum=0.9)
        scheduler_translator = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_translator,
            milestones=milestones,
            gamma=0.1,
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
                    paraphraser,
                    translator,
                    dataloaders['train'],
                    criterion,
                    optimizer,
                    optimizer_translator,
                    epoch,
                )

                valid_loss, valid_top1_acc, valid_top5_acc = self.test(
                    teacher_model,
                    student_model,
                    paraphraser,
                    translator,
                    dataloaders['val'],
                    criterion,
                    epoch,
                )

                # use conditions for different schedulers e.g. ReduceLROnPlateau needs scheduler.step(v_loss)
                scheduler.step()
                scheduler_translator.step()

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
                    f'{os.getcwd()}/logs/Factor_Transfer/{self.name}.csv',
                    index=False)

    def train_one_epoch(
        self,
        teacher_model,
        student_model,
        paraphraser,
        translator,
        dataloader,
        loss_fn,
        optimizer,
        optimizer_translator,
        epoch,
    ):
        teacher_model.eval()
        paraphraser.eval()
        student_model.train()
        translator.train()

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        end = time.time()

        epoch_iterator = tqdm(
            dataloader,
            desc=
            'Training student and translator network Epoch [X] (X / X Steps) (batch time=X.Xs) (data time=X.Xs) (loss=X.X)',
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
            feature_map_pair = [
                teacher_io_dict[self.teacher_layer_name]['output'],
                student_io_dict[self.student_layer_name]['output'],
            ]

            teacher_factor = paraphraser(feature_map_pair[0], mode=1)
            student_factor = translator(feature_map_pair[1])

            loss = loss_fn(teacher_factor, student_factor, teacher_preds,
                           student_preds, labels)
            n = images.size(0)
            losses.update(loss.item(), n)

            optimizer.zero_grad()
            optimizer_translator.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_translator.step()

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

    def test(
        self,
        teacher_model,
        student_model,
        paraphraser,
        translator,
        dataloader,
        loss_fn,
        epoch,
    ):
        teacher_model.eval()
        paraphraser.eval()
        student_model.eval()
        translator.eval()

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
                feature_map_pair = [
                    teacher_io_dict[self.teacher_layer_name]['output'],
                    student_io_dict[self.student_layer_name]['output'],
                ]

                teacher_factor = paraphraser(feature_map_pair[0], mode=1)
                student_factor = translator(feature_map_pair[1])

                loss = loss_fn(teacher_factor, student_factor, teacher_preds,
                               student_preds, labels)

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

    def criterion(self, teacher_factor, student_factor, teacher_preds,
                  student_preds, labels):
        return self.ft_loss(teacher_factor, student_factor, student_preds,
                            labels)

    def train_paraphraser(self, teacher_model, dataloaders, **kwargs):
        in_planes = kwargs.get('IN_PLANES', 64)
        rate = kwargs.get('RATE', 0.5)
        planes = kwargs.get('PLANES', int(in_planes * rate))
        paraphraser = Paraphraser(in_planes, planes)
        paraphraser.to(self.device)

        path = kwargs.get('PATH', '')
        if path != '':
            print('=====> LOADING PARAPHRASER <=====')
            logger.info('=====> LOADING PARAPHRASER <=====')
            paraphraser.load_state_dict(torch.load(path))
            self.paraphraser = paraphraser
        else:
            print('=====> TRAINING PARAPHRASER <=====')
            logger.info('=====> TRAINING PARAPHRASER <=====')
            num_epochs = kwargs.get('EPOCHS', 5)
            lr = kwargs.get('LR', 0.1)
            weight_decay = kwargs.get('WEIGHT_DECAY', 0.0005)

            optimizer = torch.optim.SGD(paraphraser.parameters(),
                                        lr=lr,
                                        weight_decay=weight_decay,
                                        momentum=0.9)
            criterion = self.l1_loss

            paraphraser.train()
            for epoch in range(num_epochs):
                t_loss = self.train_one_epoch_paraphraser(
                    teacher_model,
                    paraphraser,
                    dataloaders['train'],
                    criterion,
                    optimizer,
                    epoch,
                )

            torch.save(paraphraser.state_dict(),
                       f'checkpoints/{self.log_name}_paraphraser.pth')

            is_best = False
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': paraphraser.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                is_best,
                self.save,
                file_name='paraphraser',
            )
            self.paraphraser = paraphraser
            self.paraphraser.load_state_dict(
                torch.load(f'{self.save}/paraphraser.pth.tar')['state_dict'])

    def train_one_epoch_paraphraser(self, teacher_model, paraphraser,
                                    dataloader, criterion, optimizer, epoch):
        teacher_model.eval()
        paraphraser.train()

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        end = time.time()

        epoch_iterator = tqdm(
            dataloader,
            desc=
            'Training paraphraser network Epoch [X] (X / X Steps) (batch time=X.Xs) (data time=X.Xs) (loss=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        for i, (images, labels) in enumerate(epoch_iterator):
            data_time.update(time.time() - end)
            images = images.to(self.device, dtype=torch.float)
            labels = labels.to(self.device)

            teacher_preds = teacher_model(images)
            teacher_io_dict = self.forward_hook_manager_teacher.pop_io_dict()
            feature_map = teacher_io_dict[self.teacher_layer_name]['output']
            paraphraser_output = paraphraser(feature_map, mode=0)

            loss = criterion(paraphraser_output, feature_map.detach())
            n = images.size(0)
            losses.update(loss.item(), n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            epoch_iterator.set_description(
                'Training paraphraser Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f)'
                % (
                    epoch,
                    (i + 1),
                    len(dataloader),
                    batch_time.val,
                    data_time.val,
                    losses.val,
                ))

            logger.info(
                'Training paraphraser Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f)'
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
                    'paraphraser_train_loss': losses.val,
                })

        return losses.avg

    def register_hooks(self):
        self.forward_hook_manager_teacher.add_hook(
            self.teacher_model,
            self.teacher_layer_name,
            requires_input=False,
            requires_output=True,
        )
        self.forward_hook_manager_student.add_hook(
            self.student_model,
            self.student_layer_name,
            requires_input=False,
            requires_output=True,
        )
