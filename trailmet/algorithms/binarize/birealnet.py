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
import time
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from trailmet.algorithms.binarize.binarize import BaseBinarize
from trailmet.utils import (
    AverageMeter,
    save_checkpoint,
    accuracy,
    CrossEntropyLabelSmooth,
)

import logging
from datetime import datetime
from tqdm import tqdm
import wandb
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BirealNet(BaseBinarize):
    """
    `BirealNet <https://openaccess.thecvf.com/content_ECCV_2018/papers/zechun_liu_Bi-Real_Net_Enhancing_ECCV_2018_paper.pdf>`_ Implementation.

    References
    ----------

    Parameters
    ----------
        model (object): A pytorch model you want to use.
        dataloaders (dict): Dictionary with dataloaders for train, val and test. Keys: 'train', 'val', 'test'.
        CFG (object): YAML safe loaded file with information like batch_size, optimizer, epochs, momentum, etc.
    """

    def __init__(self, model, dataloaders, **CFG):
        self.model = model
        self.dataloaders = dataloaders
        self.CFG = CFG
        self.batch_size = self.CFG['batch_size']
        self.optimizer = self.CFG['optimizer']
        self.epochs = self.CFG['epochs']
        self.lr = self.CFG['lr']
        self.momentum = self.CFG['momentum']
        self.save_path = self.CFG['save_path']
        self.data_path = self.CFG['data_path']
        self.weight_decay = self.CFG['weight_decay']
        self.label_smooth = self.CFG['label_smooth']
        self.num_workers = self.CFG['workers']
        self.dataset = self.CFG['dataset']
        self.num_class = self.CFG['num_classes']
        self.device = self.CFG['device']

        self.wandb_monitor = self.CFG.get('wandb', 'False')
        self.dataset_name = dataloaders['train'].dataset.__class__.__name__

        self.name = '_'.join([
            self.dataset_name,
            f'{self.epochs}',
            f'{self.lr}',
            datetime.now().strftime('%b-%d_%H:%M:%S'),
        ])

        os.makedirs(f'{os.getcwd()}/logs/BiRealNet', exist_ok=True)
        self.logger_file = f'{os.getcwd()}/logs/BiRealNet/{self.name}.log'

        logging.basicConfig(
            filename=self.logger_file,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )

        logger.info(f'Experiment Arguments: {self.CFG}')

        if self.wandb_monitor:
            wandb.init(project='Trailmet BiRealNet', name=self.name)
            wandb.config.update(self.CFG)

    def train(self, epoch, train_loader, model, criterion, optimizer,
              scheduler):
        """Train function."""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        epoch_iterator = tqdm(
            train_loader,
            desc=
            'Training BiRealNet Epoch [X] (X / X Steps) (batch time=X.Xs) (data time=X.Xs) (loss=X.X) (top1=X.X) (top5=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        model.train()
        end = time.time()
        scheduler.step()

        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('learning_rate:', cur_lr)

        for i, (images, target) in enumerate(epoch_iterator):
            data_time.update(time.time() - end)
            images = images.to(device=self.device)
            target = target.to(device=self.device)

            # compute outputy
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)  # accumulated loss
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            epoch_iterator.set_description(
                'Training BiRealNet Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
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
                'Training BiRealNet Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
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

    def validate(self, epoch, val_loader, model, criterion, CFG):
        """Validate Function."""
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        epoch_iterator = tqdm(
            val_loader,
            desc=
            'Validating BiRealNet Epoch [X] (X / X Steps) (batch time=X.Xs) (loss=X.X) (top1=X.X) (top5=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        # switch to evaluation mode
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(epoch_iterator):
                images = images.to(device=self.device)
                target = target.to(device=self.device)

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
                    'Validating BiRealNet Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
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
                    'Validating BiRealNet Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
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

            print(
                ' * Val acc@1 {top1.avg:.3f} Val acc@5 {top5.avg:.3f}'.format(
                    top1=top1, top5=top5))

        return losses.avg, top1.avg, top5.avg

    def test(self, epoch, test_loader, model, criterion, CFG):
        """Test Function."""
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        epoch_iterator = tqdm(
            test_loader,
            desc=
            'Testing BiRealNet Epoch [X] (X / X Steps) (batch time=X.Xs) (loss=X.X) (top1=X.X) (top5=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        # switch to evaluation mode
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(epoch_iterator):
                images = images.to(device=self.device)
                target = target.to(device=self.device)

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
                    'Testing BiRealNet Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
                    % (
                        epoch,
                        (i + 1),
                        len(test_loader),
                        batch_time.val,
                        losses.val,
                        top1.val,
                        top5.val,
                    ))

                logger.info(
                    'Testing BiRealNet Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
                    % (
                        epoch,
                        (i + 1),
                        len(test_loader),
                        batch_time.val,
                        losses.val,
                        top1.val,
                        top5.val,
                    ))

                if self.wandb_monitor:
                    wandb.log({
                        'test_loss': losses.val,
                        'test_top1_acc': top1.val,
                        'test_top5_acc': top5.val,
                    })

            print(' * Test acc@1 {top1.avg:.3f} Test acc@5 {top5.avg:.3f}'.
                  format(top1=top1, top5=top5))

        return losses.avg, top1.avg, top5.avg

    def binarize(self):
        """Function used to binarize the model."""
        if not torch.cuda.is_available():
            sys.exit(1)
        start_t = time.time()

        cudnn.benchmark = True
        cudnn.enabled = True

        # load model
        model = self.model
        model = model.to(device=self.device)
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device=self.device)
        criterion_smooth = CrossEntropyLabelSmooth(self.num_class,
                                                   self.label_smooth)
        criterion_smooth = criterion_smooth.to(device=self.device)

        ## Preparing Optimizers
        all_parameters = model.parameters()
        weight_parameters = []
        for pname, p in model.named_parameters():
            if (p.ndimension() == 4 or pname == 'classifier.0.weight'
                    or pname == 'classifier.0.bias'):
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
            lr=self.lr,
        )
        ##################

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: (1.0 - step / self.epochs), last_epoch=-1)

        start_epoch = 0
        best_top1_acc = 0

        os.makedirs(self.save_path, exist_ok=True)
        checkpoint_tar = os.path.join(self.save_path,
                                      f'{self.dataset}-checkpoint.pth.tar')
        if os.path.exists(checkpoint_tar):
            logging.info(
                'loading checkpoint {} ..........'.format(checkpoint_tar))
            checkpoint = torch.load(checkpoint_tar)
            start_epoch = checkpoint['epoch']
            best_top1_acc = checkpoint['best_top1_acc']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info('loaded checkpoint {} epoch = {}'.format(
                checkpoint_tar, checkpoint['epoch']))

        # adjust the learning rate according to the checkpoint
        for epoch in range(start_epoch):
            scheduler.step()

        logging.info(
            'epoch, train accuracy, train loss, val accuracy, val loss')
        epoch = start_epoch
        epochs_list = []
        train_top1_acc_list = []
        train_top5_acc_list = []
        val_top1_acc_list = []
        val_top5_acc_list = []
        while epoch < self.epochs:
            train_obj, train_top1_acc, train_top5_acc = self.train(
                epoch,
                self.dataloaders['train'],
                self.model,
                criterion_smooth,
                optimizer,
                scheduler,
            )
            valid_obj, valid_top1_acc, valid_top5_acc = self.validate(
                epoch, self.dataloaders['val'], self.model, criterion,
                self.CFG)

            train_top1_acc_list.append(train_top1_acc)
            train_top5_acc_list.append(train_top5_acc)
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
                    'state_dict': model.state_dict(),
                    'best_top1_acc': best_top1_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                },
                is_best,
                self.save_path,
            )

            epoch += 1

        best = torch.load(f'{self.save_path}/model_best.pth.tar')
        self.model.load_state_dict(best['state_dict'])
        self.test((epoch - 1), self.dataloaders['test'], self.model, criterion,
                  self.CFG)
        training_time = (time.time() - start_t) / 36000
        print('total training time = {} hours'.format(training_time))

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
        df.to_csv(f'{os.getcwd()}/logs/BiRealNet/{self.name}.csv', index=False)
