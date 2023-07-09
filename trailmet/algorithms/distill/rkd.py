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
from trailmet.algorithms.distill.distill import Distillation
from trailmet.algorithms.distill.losses import KDTransferLoss, RkdDistance, RKdAngle

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


class RKDTransfer(Distillation):
    """Class to compress model using distillation via RKD transfer.

    Parameters
    ----------
        teacher_model (object): Teacher model you want to use.
        student_model (object): Student model you want to use.
        dataloaders (dict): Dictionary with dataloaders for train, val and test. Keys: 'train', 'val', 'test'.
        paraphraser (object): Paraphrase model
        kwargs (object): YAML safe loaded file with information like device, distill_args(beta, temperature, etc).
    """

    def __init__(self, teacher_model, student_model, dataloaders, **kwargs):
        super(RKDTransfer, self).__init__(**kwargs)
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.dataloaders = dataloaders
        self.kwargs = kwargs
        self.device = kwargs['DEVICE']
        self.lambda_ = self.kwargs['DISTILL_ARGS'].get('BETA', {
            'lambda_d': 25,
            'lambda_a': 50
        })

        # self.student_io_dict, self.teacher_io_dict = dict(), dict()
        self.temperature = self.kwargs['DISTILL_ARGS'].get('TEMPERATURE', 5)
        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = KDTransferLoss(self.temperature)
        self.distance_loss = RkdDistance()
        self.angle_loss = RKdAngle()

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

        os.makedirs(f'{os.getcwd()}/logs/Relation_KD', exist_ok=True)
        os.makedirs(self.save, exist_ok=True)
        self.logger_file = f'{os.getcwd()}/logs/Relation_KD/{self.name}.log'

        logging.basicConfig(
            filename=self.logger_file,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )

        logger.info(f'Experiment Arguments: {self.kwargs}')

        if self.wandb_monitor:
            wandb.init(project='Trailmet Relation_KD', name=self.name)
            wandb.config.update(self.kwargs)

    def compress_model(self):
        """Function to transfer knowledge from teacher model to student
        model."""
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
        # self.register_hooks()
        test_only = kwargs.get('TEST_ONLY', False)
        weight_decay = kwargs.get('WEIGHT_DECAY', 0.0005)

        # dont hard code this
        optimizer = torch.optim.SGD(
            student_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[60, 120, 160], gamma=0.2, verbose=False)
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

            teacher_pool, teacher_preds = teacher_model.forward(images,
                                                                is_feat=True)
            student_pool, student_preds = student_model.forward(images,
                                                                is_feat=True)

            feature_map_pairs = []
            feature_map_pairs.append((teacher_pool, student_pool))

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
        teacher_model.eval()
        student_model.eval()

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

                with torch.no_grad():
                    teacher_pool, teacher_preds = teacher_model.forward(
                        images, is_feat=True)
                    student_pool, student_preds = student_model.forward(
                        images, is_feat=True)

                feature_map_pairs = []
                feature_map_pairs.append((teacher_pool, student_pool))
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
        kd_loss = self.kd_loss(teacher_preds, student_preds)
        teacher_pool = feature_map_pairs[0][0]
        student_pool = feature_map_pairs[0][1]
        # teacher_pool=teacher_pool.view(teacher_pool.size(0),-1)
        # student_pool=student_pool.view(student_pool.size(0),-1)
        distance_loss = self.distance_loss(student_pool, teacher_pool)
        angle_loss = self.angle_loss(student_pool, teacher_pool)
        lambda_d = self.lambda_['lambda_d']
        lambda_a = self.lambda_['lambda_a']

        return (ce_loss + (self.temperature**2) * kd_loss +
                lambda_d * distance_loss + lambda_a * angle_loss)
