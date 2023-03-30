# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
sys.path.append('/workspace/code/kushagrabhushan/TrAIL/trailmet/trailmet/algorithms/binarize')
sys.path.append("../../../")
from .utils import *
from trailmet.algorithms.binarize.binarize import BaseBinarize
from trailmet.utils import *


class BirealNet(BaseBinarize):
    def __init__(self, model, dataloaders, **CFG):
        self.model = model
        self.dataloaders = dataloaders
        self.CFG = CFG
        self.batch_size = self.CFG['batch_size']
        self.optimizer = eval(self.CFG['optimizer'])
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
    
    def prepare_dirs(self):
        if not os.path.exists('log'):
            print('Creating Logging Directory...')
            os.mkdir('log')
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join('log/log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        if not os.path.exists(self.save_path):
            print('Creating Checkpoint Directory...')
            os.mkdir(self.save_path)
    
    def train(self, epoch, train_loader, model, criterion, optimizer, scheduler):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        model.train()
        end = time.time()
        scheduler.step()

        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('learning_rate:', cur_lr)

        for i, (images, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            images = images.to(device=self.device)
            target = target.to(device=self.device)

            # compute outputy
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)   #accumulated loss
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)
        logging.info
        return losses.avg, top1.avg, top5.avg

    def validate(self, epoch, val_loader, model, criterion, CFG):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        # switch to evaluation mode
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
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

                progress.display(i)

            print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

        return losses.avg, top1.avg, top5.avg

    def test(self, epoch, test_loader, model, criterion, CFG):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(test_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        # switch to evaluation mode
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(test_loader):
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

                progress.display(i)

            print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

        return losses.avg, top1.avg, top5.avg
    
    def binarize(self):
        self.prepare_dirs()
        if not torch.cuda.is_available():
            sys.exit(1)
        start_t = time.time()

        cudnn.benchmark = True
        cudnn.enabled=True
        logging.info("CFG = %s", self.CFG)
        
        # load model
        model = self.model
#         model = nn.DataParallel(model).to(device=self.device)
        model = model.to(device=self.device)
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device=self.device)
        criterion_smooth = CrossEntropyLabelSmooth(self.num_class, self.label_smooth)
        criterion_smooth = criterion_smooth.to(device=self.device)
        
        ## Preparing Optimizers
        all_parameters = model.parameters()
        weight_parameters = []
        for pname, p in model.named_parameters():
            if p.ndimension() == 4 or pname=='classifier.0.weight' or pname == 'classifier.0.bias':
                weight_parameters.append(p)
        weight_parameters_id = list(map(id, weight_parameters))
        other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

        optimizer = self.optimizer(
                [{'params' : other_parameters},
                {'params' : weight_parameters, 'weight_decay' :self.weight_decay}],
                lr=self.lr,)
        ##################
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/self.epochs), last_epoch=-1)
        
        start_epoch = 0
        best_top1_acc= 0

        checkpoint_tar = os.path.join(self.save_path, f'{self.dataset}-checkpoint.pth.tar')
        if os.path.exists(checkpoint_tar):
            logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
            checkpoint = torch.load(checkpoint_tar)
            start_epoch = checkpoint['epoch']
            best_top1_acc = checkpoint['best_top1_acc']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))

        # adjust the learning rate according to the checkpoint
        for epoch in range(start_epoch):
            scheduler.step()
        
        logging.info('epoch, train accuracy, train loss, val accuracy, val loss')
        epoch = start_epoch
        while epoch < self.epochs:
            train_obj, train_top1_acc,  train_top5_acc = self.train(epoch,  self.dataloaders['train'], self.model, criterion_smooth, optimizer, scheduler)
            valid_obj, valid_top1_acc, valid_top5_acc = self.validate(epoch, self.dataloaders['val'], self.model, criterion, self.CFG)
            logging.info("{}, {}, {}, {}, {}".format(epoch, train_top1_acc, train_obj, valid_top1_acc.item(), valid_obj))
            is_best = False
            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                is_best = True

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_top1_acc': best_top1_acc,
                'optimizer' : optimizer.state_dict(),
                }, is_best, self.save_path)

            epoch += 1
        
        best = torch.load(f"{self.save_path}/model_best.pth.tar")
        self.model.load_state_dict(best['state_dict'])
        self.test(epoch, self.dataloaders['test'], self.model, criterion, self.CFG)
        training_time = (time.time() - start_t) / 36000
        print('total training time = {} hours'.format(training_time))



