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
import torch.utils
import torch.backends.cudnn as cudnn
import math
import shutil
from pathlib import Path
from collections import OrderedDict
from thop import profile
import time, datetime
import logging
import torch.utils.data.distributed
from torch.cuda.amp import autocast, GradScaler
from trailmet.algorithms.prune.prune import BasePruning
from trailmet.models.resnet_chip import resnet_50

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
    CrossEntropyLabelSmooth,
    extract_sparsity,
    chip_adjust_learning_rate,
)

logger = logging.getLogger(__name__)


class Chip(BasePruning):
    """
    Parameters
    ----------
        model (object): A pytorch model you want to use.
        dataloaders (dict): Dictionary with dataloaders for train, val and test. Keys: 'train', 'val', 'test'.
        CFG (object): YAML safe loaded file with information like batch_size, arch, epochs, momentum, etc.
    """

    def __init__(self, model, dataloaders, **CFG):
        self.dataloaders = dataloaders
        self.model = model
        self.CFG = CFG
        self.batch_size = self.CFG['batch_size']
        self.arch = self.CFG['arch']
        self.repeat = self.CFG['repeat']
        self.ci_dir = self.CFG['ci_dir']
        self.lr_type = self.CFG['lr_type']
        self.learning_rate = self.CFG['learning_rate']
        self.epochs = self.CFG['epochs']
        self.num_layers = self.CFG['num_layers']
        self.feature_map_dir = self.CFG['feature_map_dir']
        self.sparsity = self.CFG['sparsity']
        self.label_smooth = self.CFG['label_smooth']
        self.device = self.CFG['device']
        self.gpu = self.CFG['gpu']
        self.momentum = self.CFG['momentum']
        self.weight_decay = self.CFG['weight_decay']
        self.lr_decay_step = self.CFG['lr_decay_step']
        self.result_dir = self.CFG['result_dir']
        self.pretrain_dir = self.CFG['pretrain_dir']
        self.conv_index = torch.tensor([self.CFG['conv_index']])

        self.wandb_monitor = self.CFG.get('wandb', 'False')
        self.dataset_name = dataloaders['train'].dataset.__class__.__name__
        self.save = './checkpoints/'

        self.name = '_'.join([
            self.dataset_name,
            f'{self.epochs}',
            f'{self.learning_rate}',
            datetime.now().strftime('%b-%d_%H:%M:%S'),
        ])

        os.makedirs(f'{os.getcwd()}/logs/Chip', exist_ok=True)
        os.makedirs(self.save, exist_ok=True)
        self.logger_file = f'{os.getcwd()}/logs/Chip/{self.name}.log'

        logging.basicConfig(
            filename=self.logger_file,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )

        logger.info(f'Experiment Arguments: {self.CFG}')

        if self.wandb_monitor:
            wandb.init(project='Trailmet Chip', name=self.name)
            wandb.config.update(self.CFG)

    def get_feature_hook(self, module, input, output):
        os.makedirs('conv_feature_map/' + self.arch + '_repeat%d' %
                    (self.repeat),
                    exist_ok=True)
        np.save(
            'conv_feature_map/' + self.arch + '_repeat%d' % (self.repeat) +
            '/conv_feature_map_' + str(self.conv_index) + '.npy',
            output.cpu().numpy(),
        )
        self.conv_index += 1

    def inference(self):
        model = self.model
        model.eval()
        model.to(self.device)
        repeat = self.repeat
        with torch.no_grad():
            for batch_idx, (inputs,
                            targets) in enumerate(self.dataloaders['train']):
                # use 5 batches to get feature maps.
                if batch_idx >= repeat:
                    break

                inputs, targets = inputs.to(self.device), targets.to(
                    self.device)

                model(inputs)

    def reduced_1_row_norm(self, input, row_index, data_index):
        input[data_index, row_index, :] = torch.zeros(input.shape[-1])
        m = torch.norm(input[data_index, :, :], p='nuc').item()
        return m

    def ci_score(self, path_conv):
        conv_output = torch.tensor(np.round(np.load(path_conv), 4))
        conv_reshape = conv_output.reshape(conv_output.shape[0],
                                           conv_output.shape[1], -1)

        r1_norm = torch.zeros([conv_reshape.shape[0], conv_reshape.shape[1]])
        for i in range(conv_reshape.shape[0]):
            for j in range(conv_reshape.shape[1]):
                r1_norm[i, j] = self.reduced_1_row_norm(conv_reshape.clone(),
                                                        j,
                                                        data_index=i)

        ci = np.zeros_like(r1_norm)

        for i in range(r1_norm.shape[0]):
            original_norm = torch.norm(torch.tensor(conv_reshape[i, :, :]),
                                       p='nuc').item()
            ci[i] = original_norm - r1_norm[i]

        # return shape: [batch_size, filter_number]
        return ci

    def mean_repeat_ci(self, repeat, num_layers):
        layer_ci_mean_total = []
        for j in range(num_layers):
            print(f'Caclulating Mean Repeat CI for layer {j}')
            logger.info(f'Caclulating Mean Repeat CI for layer {j}')
            repeat_ci_mean = []
            for i in tqdm(range(repeat),
                          desc=f'Repeat CI for layer {j}',
                          total=repeat):
                index = j * repeat + i + 1
                # add
                path_conv = './conv_feature_map/{0}_repeat5/conv_feature_map_tensor({1}).npy'.format(
                    str(self.arch), str(index))

                batch_ci = self.ci_score(path_conv)
                single_repeat_ci_mean = np.mean(batch_ci, axis=0)
                repeat_ci_mean.append(single_repeat_ci_mean)

            layer_ci_mean = np.mean(repeat_ci_mean, axis=0)
            layer_ci_mean_total.append(layer_ci_mean)

        return np.array(layer_ci_mean_total)

    def load_resnet_model(self, model, oristate_dict):
        if len(self.gpu) > 1:
            name_base = 'module.'
        else:
            name_base = ''
        cfg = {
            'resnet_50': [3, 4, 6, 3],
        }

        state_dict = model.state_dict()

        current_cfg = cfg[self.arch]
        last_select_index = None

        all_honey_conv_weight = []

        bn_part_name = [
            '.weight',
            '.bias',
            '.running_mean',
            '.running_var',
        ]  # ,'.num_batches_tracked']
        prefix = self.ci_dir + '/ci_conv'
        subfix = '.npy'
        cnt = 1

        conv_weight_name = 'conv1.weight'
        all_honey_conv_weight.append(conv_weight_name)
        oriweight = oristate_dict[conv_weight_name]
        curweight = state_dict[name_base + conv_weight_name]
        orifilter_num = oriweight.size(0)
        currentfilter_num = curweight.size(0)

        if orifilter_num != currentfilter_num:
            logger.info('loading ci from: ' + prefix + str(cnt) + subfix)
            ci = np.load(prefix + str(cnt) + subfix)
            select_index = np.argsort(ci)[
                orifilter_num - currentfilter_num:]  # preserved filter id
            select_index.sort()

            for index_i, i in enumerate(select_index):
                state_dict[name_base + conv_weight_name][
                    index_i] = oristate_dict[conv_weight_name][i]
                for bn_part in bn_part_name:
                    state_dict[name_base + 'bn1' +
                               bn_part][index_i] = oristate_dict['bn1' +
                                                                 bn_part][i]

            last_select_index = select_index
        else:
            state_dict[name_base + conv_weight_name] = oriweight
            for bn_part in bn_part_name:
                state_dict[name_base + 'bn1' +
                           bn_part] = oristate_dict['bn1' + bn_part]

        state_dict[name_base + 'bn1' + '.num_batches_tracked'] = oristate_dict[
            'bn1' + '.num_batches_tracked']

        cnt += 1
        for layer, num in enumerate(current_cfg):
            layer_name = 'layer' + str(layer + 1) + '.'

            for k in range(num):
                iter = 3
                if k == 0:
                    iter += 1
                for l in range(iter):
                    record_last = True
                    if k == 0 and l == 2:
                        conv_name = layer_name + str(k) + '.downsample.0'
                        bn_name = layer_name + str(k) + '.downsample.1'
                        record_last = False
                    elif k == 0 and l == 3:
                        conv_name = layer_name + str(k) + '.conv' + str(l)
                        bn_name = layer_name + str(k) + '.bn' + str(l)
                    else:
                        conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                        bn_name = layer_name + str(k) + '.bn' + str(l + 1)

                    conv_weight_name = conv_name + '.weight'
                    all_honey_conv_weight.append(conv_weight_name)
                    oriweight = oristate_dict[conv_weight_name]
                    curweight = state_dict[name_base + conv_weight_name]
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)

                    if orifilter_num != currentfilter_num:
                        logger.info('loading ci from: ' + prefix + str(cnt) +
                                    subfix)
                        ci = np.load(prefix + str(cnt) + subfix)
                        select_index = np.argsort(
                            ci)[orifilter_num -
                                currentfilter_num:]  # preserved filter id
                        select_index.sort()

                        if last_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_select_index):
                                    state_dict[name_base + conv_weight_name][
                                        index_i][index_j] = oristate_dict[
                                            conv_weight_name][i][j]

                                for bn_part in bn_part_name:
                                    state_dict[name_base + bn_name + bn_part][
                                        index_i] = oristate_dict[bn_name +
                                                                 bn_part][i]

                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[
                                    name_base +
                                    conv_weight_name][index_i] = oristate_dict[
                                        conv_weight_name][i]

                                for bn_part in bn_part_name:
                                    state_dict[name_base + bn_name + bn_part][
                                        index_i] = oristate_dict[bn_name +
                                                                 bn_part][i]

                        if record_last:
                            last_select_index = select_index

                    elif last_select_index is not None:
                        for index_i in range(orifilter_num):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base + conv_weight_name][
                                    index_i][index_j] = oristate_dict[
                                        conv_weight_name][index_i][j]

                        for bn_part in bn_part_name:
                            state_dict[name_base + bn_name +
                                       bn_part] = oristate_dict[bn_name +
                                                                bn_part]

                        if record_last:
                            last_select_index = None

                    else:
                        state_dict[name_base + conv_weight_name] = oriweight
                        for bn_part in bn_part_name:
                            state_dict[name_base + bn_name +
                                       bn_part] = oristate_dict[bn_name +
                                                                bn_part]
                        if record_last:
                            last_select_index = None

                    state_dict[name_base + bn_name +
                               '.num_batches_tracked'] = oristate_dict[
                                   bn_name + '.num_batches_tracked']
                    cnt += 1

        for name, module in model.named_modules():
            name = name.replace('module.', '')
            if isinstance(module, nn.Conv2d):
                conv_name = name + '.weight'
                if conv_name not in all_honey_conv_weight:
                    state_dict[name_base +
                               conv_name] = oristate_dict[conv_name]

            elif isinstance(module, nn.Linear):
                state_dict[name_base + name +
                           '.weight'] = oristate_dict[name + '.weight']
                state_dict[name_base + name + '.bias'] = oristate_dict[name +
                                                                       '.bias']

        model.load_state_dict(state_dict)

    def compress_model(self):
        model = self.model
        cov_layer = eval('model.maxpool')
        handler = cov_layer.register_forward_hook(self.get_feature_hook)
        self.inference()
        handler.remove()

        # ResNet50 per bottleneck
        for i in range(4):
            print(f'Calculating feature maps of ResNet50 for block {i}')
            logger.info(f'Calculating feature maps of ResNet50 for block {i}')
            block = eval('model.layer%d' % (i + 1))
            for j in tqdm(range(model.num_blocks[i]),
                          desc=f'Block {i}',
                          total=model.num_blocks[i]):
                cov_layer = block[j].relu1
                handler = cov_layer.register_forward_hook(
                    self.get_feature_hook)
                self.inference()
                handler.remove()

                cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(
                    self.get_feature_hook)
                self.inference()
                handler.remove()

                cov_layer = block[j].relu3
                handler = cov_layer.register_forward_hook(
                    self.get_feature_hook)
                self.inference()
                handler.remove()

                if j == 0:
                    cov_layer = block[j].relu3
                    handler = cov_layer.register_forward_hook(
                        self.get_feature_hook)
                    self.inference()
                    handler.remove()

        repeat = self.repeat
        num_layers = self.num_layers
        save_path = os.path.join(self.save, 'CI_' + self.arch)
        ci = self.mean_repeat_ci(repeat, num_layers)

        if self.arch == 'resnet_50':
            num_layers = 53

        for i in range(num_layers):
            print(f'Saving CI of layer {i}')
            logger.info(f'Saving CI of layer {i}')
            os.makedirs(save_path, exist_ok=True)
            np.save(save_path + '/ci_conv{0}.npy'.format(str(i + 1)), ci[i])

        os.makedirs(self.result_dir, exist_ok=True)

        # save old training file
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        cp_file_dir = os.path.join(self.result_dir, 'cp_file/' + now)
        if os.path.exists(self.result_dir + '/model_best.pth.tar'):
            if not os.path.isdir(cp_file_dir):
                os.makedirs(cp_file_dir)
            shutil.copy(self.result_dir + '/config.txt', cp_file_dir)
            shutil.copy(self.result_dir + '/logger.log', cp_file_dir)
            shutil.copy(self.result_dir + '/model_best.pth.tar', cp_file_dir)
            shutil.copy(self.result_dir + '/checkpoint.pth.tar', cp_file_dir)

        cudnn.benchmark = True
        cudnn.enabled = True
        logger.info('args = %s', self)

        if self.sparsity:
            sparsity = extract_sparsity(self.sparsity)

        # load model
        logger.info('sparsity:' + str(sparsity))
        logger.info('==> Building model..')
        model = eval(self.arch)(sparsity=sparsity).cuda()
        logger.info(model)

        CLASSES = self.dataloaders['train'].dataset.classes
        print_freq = 128000 // self.batch_size
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        criterion_smooth = CrossEntropyLabelSmooth(CLASSES, self.label_smooth)
        criterion_smooth = criterion_smooth.cuda()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        lr_decay_step = list(map(int, self.lr_decay_step.split(',')))
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

        start_epoch = 0
        best_top1_acc = 0
        best_top5_acc = 0

        # load the checkpoint if it exists
        checkpoint_dir = os.path.join(self.result_dir, 'checkpoint.pth.tar')

        logger.info('resuming from pretrain model')
        origin_model = eval(self.arch)(sparsity=[0.0] * 100).cuda()
        ckpt = torch.load(self.pretrain_dir)
        origin_model.load_state_dict(ckpt)
        oristate_dict = origin_model.state_dict()
        if self.arch == 'resnet_50':
            self.load_resnet_model(model, oristate_dict)
        else:
            raise

        # adjust the learning rate according to the checkpoint
        # for epoch in range(start_epoch):
        #     scheduler.step()

        # train the model
        scaler = GradScaler()
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
                model,
                criterion_smooth,
                optimizer,
                scaler,
            )
            valid_obj, valid_top1_acc, valid_top5_acc = self.validate(
                epoch, self.dataloaders['val'], model, criterion)

            is_best = False
            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                best_top5_acc = valid_top5_acc
                is_best = True

            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_top1_acc': best_top1_acc,
                    'best_top5_acc': best_top5_acc,
                    'optimizer': optimizer.state_dict(),
                },
                is_best,
                self.result_dir,
            )

            epoch += 1

            if self.wandb_monitor:
                wandb.log({'best_top1_acc': best_top1_acc})

            epochs_list.append(epoch)
            train_top1_acc_list.append(train_top1_acc)
            train_top5_acc_list.append(train_top5_acc)
            val_top1_acc_list.append(valid_top1_acc.cpu().numpy())
            val_top5_acc_list.append(valid_top5_acc.cpu().numpy())

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

            df.to_csv(
                f'{os.getcwd()}/logs/Chip/{self.name}.csv',
                index=False,
            )

    def train(self,
              epoch,
              train_loader,
              model,
              criterion,
              optimizer,
              scaler=None):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        model.train()
        end = time.time()

        epoch_iterator = tqdm(
            train_loader,
            desc=
            'Training network Epoch [X] (X / X Steps) (batch time=X.Xs) (data time=X.Xs) (loss=X.X) (top1=X.X) (top5=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        for i, (images, targets) in enumerate(epoch_iterator):
            images = images.cuda()
            targets = targets.cuda()
            data_time.update(time.time() - end)

            chip_adjust_learning_rate(optimizer, epoch, i, len(train_loader))

            # compute output
            logits = model(images)
            loss = criterion(logits, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
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
                'Training network Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
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
                'Training network Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
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
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        epoch_iterator = tqdm(
            val_loader,
            desc=
            'Validating network Epoch [X] (X / X Steps) (batch time=X.Xs) (loss=X.X) (top1=X.X) (top5=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (images, targets) in enumerate(epoch_iterator):
                images = images.cuda()
                targets = targets.cuda()

                # compute output
                logits = model(images)
                loss = criterion(logits, targets)

                # measure accuracy and record loss
                pred1, pred5 = accuracy(logits, targets, topk=(1, 5))
                n = images.size(0)
                losses.update(loss.item(), n)
                top1.update(pred1[0], n)
                top5.update(pred5[0], n)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                epoch_iterator.set_description(
                    'Validating network Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
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
                    'Validating network Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
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

            logger.info(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'.format(
                top1=top1, top5=top5))

        return losses.avg, top1.avg, top5.avg
