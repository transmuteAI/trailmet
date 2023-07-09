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
import torch.utils.data
import torchvision.models as models

from trailmet.algorithms.prune.pruner import pruner_dict
from trailmet.algorithms.prune.prune import BasePruning

import logging
from datetime import datetime
from tqdm import tqdm
import wandb
import pandas as pd
import numpy as np
import os
import time

from trailmet.utils import (
    AverageMeter,
    accuracy,
    save_checkpoint,
    strlist_to_list,
    adjust_learning_rate,
)

logger = logging.getLogger(__name__)


def is_single_branch(name):
    return False


class Growth_Regularisation(BasePruning):
    """Base Algorithm class that defines the structure of each model
    compression algorithm implemented in this library.

    Every new algorithm is expected to directly use or overwrite the template
    functions defined below. The root command to invoke the compression of any
    model is .compress_model(). Thus, it is required that all algorithms
    complete this template function and use it as the first point of invoking
    the model compression process. For methods that require to perform
    pretraining and fine-tuning, the implementation of base_train() method can
    directly be used for both the tasks. In case of modifications, overwrite
    this function based on the needs.

     Parameters
    ----------
        model (object): A pytorch model you want to use.
        dataloaders (dict): Dictionary with dataloaders for train, val and test. Keys: 'train', 'val', 'test'.
        kwargs (object): YAML safe loaded file with information like dataset, num_classes, epoch, weight_decay, etc.
    """

    def __init__(self, model, dataloaders, **kwargs):
        self.device = 'cuda'
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.test_loader = dataloaders['test']
        self.dataloaders = dataloaders
        self.model = model
        self.kwargs = kwargs
        self.dataset = self.kwargs.get('DATASET', 'c100')
        self.num_classes = self.kwargs.get('num_classes', 100)
        self.label_smooth = self.kwargs.get('label_smooth', 0.1)
        self.pretrained = self.kwargs.get('pretrained', '')
        self.resume = self.kwargs.get('resume', 'False')
        self.epochs = self.kwargs.get('epoch', '120')
        self.weight_decay = self.kwargs.get('weight_decay', 0)
        self.learning_rate = self.kwargs.get('learning_rate', 0.001)
        self.stage_pr = self.kwargs.get('stage_pr', None)
        self.skip_layers = self.kwargs.get('skip_layers', None)
        self.base_pr_model = self.kwargs.get('base_pr_model', None)
        self.base_model_path = self.kwargs.get('base_model_path', None)
        self.momentum = self.kwargs.get('momentum', 0.9)
        self.save = self.kwargs.get('save', './checkpoint')
        self.method = self.kwargs.get('method', 'GReg-1')

        self.wandb_monitor = self.kwargs.get('wandb', 'False')
        self.dataset_name = dataloaders['train'].dataset.__class__.__name__

        self.name = '_'.join([
            self.dataset_name,
            f'{self.epochs}',
            f'{self.learning_rate}',
            datetime.now().strftime('%b-%d_%H:%M:%S'),
        ])

        os.makedirs(f'{os.getcwd()}/logs/Growth_Regularisation', exist_ok=True)
        os.makedirs(self.save, exist_ok=True)
        self.logger_file = f'{os.getcwd()}/logs/Growth_Regularisation/{self.name}.log'

        logging.basicConfig(
            filename=self.logger_file,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )

        logger.info(f'Experiment Arguments: {self.kwargs}')

        if self.wandb_monitor:
            wandb.init(project='Trailmet Growth_Regularisation',
                       name=self.name)
            wandb.config.update(self.kwargs)

        if self.stage_pr:
            self.stage_pr = self.stage_pr  # example: [0, 0.4, 0.5, 0]
            self.skip_layers = strlist_to_list(self.skip_layers,
                                               str)  # example: [2.3.1, 3.1]
        else:
            assert (
                self.base_pr_model
            ), 'If stage_pr is not provided, base_pr_model must be provided'

    def compress_model(self) -> None:
        """Template function to be overwritten for each model compression
        method."""

        self.epochs = 1

        self.model.maxpool = torch.nn.Identity()
        self.model = self.base_train(self.model,
                                     self.dataloaders,
                                     fine_tune=False)

        self.prune_and_finetune(self.kwargs, self.dataloaders)

    def prune_and_finetune(self, args, dataloader):
        """Prune and finetune the model."""

        # if self.base_model_path != None:
        if 0:
            X = torch.load(self.base_model_path)
            X1 = X['state_dict']
            L = list(X1.keys())
            for key in L:
                new_key = key.replace('model.', '')
                X1[new_key] = X1.pop(key)
            self.model.load_state_dict(X1)
            print("==> Load pretrained model successfully: '%s'" %
                  self.base_model_path)
            logger.info("==> Load pretrained model successfully: '%s'" %
                        self.base_model_path)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        prune_state, pruner = '', None
        if prune_state != 'finetune':

            class passer:
                pass  # to pass arguments

            passer.train_loader = dataloader['train']
            passer.test_loader = dataloader['val']
            passer.save = save_checkpoint
            passer.criterion = criterion
            passer.train_sampler = None
            passer.pruner = pruner
            passer.args = self.kwargs
            passer.is_single_branch = is_single_branch
            pruner = pruner_dict[self.method](self.model, self.kwargs, logger,
                                              passer)
            if self.method == 'L1':
                model = pruner.prune()
                print('==> Saving model without key <==')
                print(model)
                save_checkpoint(
                    {
                        'arch': args.arch,
                        'model': model,
                        'state_dict': model.state_dict(),
                    },
                    False,
                    self.save,
                )
            else:
                pruning_key, model = pruner.prune()  # get the pruned model
                print('==> Saving model with key <==')
                print(model)
                save_checkpoint(
                    {
                        'arch': args.arch,
                        'model': model,
                        'state_dict': model.state_dict(),
                        'pruning_key': pruning_key,
                    },
                    False,
                    self.save,
                )
        self.base_train(model, dataloader, pruning_key)

    def base_train(self, model, dataloaders, fine_tune=False):
        """This function is used to perform standard model training and can be
        used for various purposes, such as model pretraining, fine-tuning of
        compressed models, among others.

        For cases, where base_train is not directly applicable, feel free to
        overwrite wherever this parent class is inherited.
        """
        best_top1_acc = 0  # setting to lowest possible value
        scheduler_type = 1
        self.pr = None
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        ###########################

        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        epochs_list = []
        val_top1_acc_list = []
        val_top5_acc_list = []

        for epoch in range(self.epochs):
            adjust_learning_rate(optimizer, epoch, self.epochs, scheduler_type,
                                 self.learning_rate)

            t_loss = self.train_one_epoch(
                model,
                dataloaders['train'],
                criterion,
                optimizer,
                epoch,
            )

            valid_loss, valid_top1_acc, valid_top5_acc = self.test(
                model,
                dataloaders['val'],
                criterion,
                epoch,
            )

            is_best = False
            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                is_best = True

                if self.pr is not None:
                    print('==> Saving model with key <==')
                    logger.info('==> Saving model with key <==')
                    save_checkpoint(
                        {
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'best_top1_acc': best_top1_acc,
                            'pruning_key': self.pr,
                            'optimizer': optimizer.state_dict(),
                        },
                        is_best,
                        self.save,
                    )

                else:
                    print('==> Saving model without key <==')
                    logger.info('==> Saving model without key <==')
                    save_checkpoint(
                        {
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'best_top1_acc': best_top1_acc,
                            'pruning_key': self.pr,
                            'optimizer': optimizer.state_dict(),
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
                    f'{os.getcwd()}/logs/Growth_Regularisation/{self.name}.csv',
                    index=False,
                )

        return model

    def train_one_epoch(
        self,
        model,
        dataloader,
        loss_fn,
        optimizer,
        epoch,
        extra_functionality=None,
    ):
        """Standard training loop which can be used for various purposes with
        an extra functionality function to add to its working at the end of the
        loop."""
        model.train()

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        end = time.time()

        epoch_iterator = tqdm(
            dataloader,
            desc=
            'Training Epoch [X] (X / X Steps) (batch time=X.Xs) (data time=X.Xs) (loss=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        for i, (images, labels) in enumerate(epoch_iterator):
            data_time.update(time.time() - end)
            images = images.to(device=self.device)
            labels = labels.to(device=self.device)
            scores = model(images)

            loss = loss_fn(scores, labels)
            n = images.size(0)
            losses.update(loss.item(), n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            epoch_iterator.set_description(
                'Training Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f)'
                % (
                    epoch,
                    (i + 1),
                    len(dataloader),
                    batch_time.val,
                    data_time.val,
                    losses.val,
                ))

            logger.info(
                'Training Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f)'
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
                    f'train_loss': losses.val,
                })

            if extra_functionality is not None:
                extra_functionality()

        return losses.avg

    def test(self, model, dataloader, loss_fn=None, epoch=0):
        """This method is used to test the performance of the trained model."""

        model.eval()
        model.to(self.device)

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        epoch_iterator = tqdm(
            dataloader,
            desc=
            'Validating Epoch [X] (X / X Steps) (batch time=X.Xs) (loss=X.X) (top1=X.X) (top5=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        with torch.no_grad():
            end = time.time()
            for i, (images, targets) in enumerate(epoch_iterator):
                images = images.to(self.device)
                targets = targets.to(self.device)
                outputs = model(images)
                pred1, pred5 = accuracy(outputs, targets, topk=(1, 5))

                if loss_fn is not None:
                    loss = loss_fn(outputs, targets)

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
                        len(dataloader),
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
                        len(dataloader),
                        batch_time.val,
                        losses.val,
                        top1.val,
                        top5.val,
                    ))

                if self.wandb_monitor:
                    wandb.log({
                        f'val_loss': losses.val,
                        f'val_top1_acc': top1.val,
                        f'val_top5_acc': top5.val,
                    })

            print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'.format(
                top1=top1, top5=top5))

        return losses.avg, top1.avg, top5.avg
