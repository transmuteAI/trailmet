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

import timm
from .utils import *
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.models as models
from timm.utils.agc import adaptive_clip_grad

class BNNBN():
    def __init__(self, model, dataloaders, **kwargs):
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.test_loader = dataloaders['test']
        self.model_student = model 
        self.kwargs = kwargs
        self.dataset = self.kwargs.get('DATASET', 'c100')
        self.num_classes = self.kwargs.get('num_classes',100)
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
        
    def compress_model(self):
        self.prepare_dirs()
        if not torch.cuda.is_available():
            sys.exit(1)
        start_t = time.time()

        cudnn.benchmark = True
        cudnn.enabled=True
        
        # self.model_student = nn.DataParallel(self.model_student).cuda()
        self.model_student.cuda()
        print(self.model_student)

        # load teacher model
        if self.loss_type == 'kd':
            print('* Loading teacher model')
            if not 'nfnet' in self.teacher:
                model_teacher = models.__dict__[self.teacher](pretrained=True)
                classes_in_teacher = model_teacher.fc.out_features
                num_features = model_teacher.fc.in_features
            else:
                model_teacher = timm.create_model(self.teacher, pretrained=True)
                classes_in_teacher = model_teacher.head.fc.out_features
                num_features = model_teacher.head.fc.in_features

            if not classes_in_teacher == self.num_classes:
                print('* change fc layers in teacher')
                if not 'nfnet' in self.teacher:
                    model_teacher.fc = nn.Linear(num_features, self.num_classes)
                else:
                    model_teacher.head.fc = nn.Linear(num_features, self.num_classes)
                print('* loading pretrained teacher weight from {}'.format(self.teacher_weight))
                pretrain_teacher = torch.load(self.teacher_weight, map_location='cpu')['state_dict']
                model_teacher.load_state_dict(pretrain_teacher)

#             model_teacher = nn.DataParallel(model_teacher).cuda()
            model_teacher.cuda()
            for p in model_teacher.parameters():
                p.requires_grad = False
            model_teacher.eval()


        #criterion
        criterion = nn.CrossEntropyLoss().cuda()
        criterion_smooth = CrossEntropyLabelSmooth(self.num_classes, self.label_smooth).cuda()
        criterion_kd = DistributionLoss()

        #optimizer
        all_parameters = self.model_student.parameters()
        weight_parameters = []
        for pname, p in self.model_student.named_parameters():
            if p.ndimension() == 4 or 'conv' in pname:
                weight_parameters.append(p)
        weight_parameters_id = list(map(id, weight_parameters))
        other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

        optimizer = torch.optim.Adam(
                [{'params' : other_parameters},
                {'params' : weight_parameters, 'weight_decay' : self.weight_decay}],
                lr=self.learning_rate,)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/self.epochs), last_epoch=-1)
        start_epoch = 0
        best_top1_acc= 0

        if self.pretrained:
            print('* loading pretrained weight {}'.format(self.pretrained))
            logging.info(f'loading pretrained weight {self.pretrained}')
            pretrain_student = torch.load(args.pretrained)
            if 'state_dict' in pretrain_student.keys():
                pretrain_student = pretrain_student['state_dict']

            for key in pretrain_student.keys():
                if not key in self.model_student.state_dict().keys():
                    print('unload key: {}'.format(key))

            self.model_student.load_state_dict(pretrain_student, strict=False)

        if self.resume:
            checkpoint_tar = os.path.join(self.save, 'checkpoint.pth.tar')
            if os.path.exists(checkpoint_tar):
                print('loading checkpoint {} ..........'.format(checkpoint_tar))
                logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
                checkpoint = torch.load(checkpoint_tar)
                start_epoch = checkpoint['epoch']
                best_top1_acc = checkpoint['best_top1_acc']
                self.model_student.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))
                logging.info("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))
            else:
                raise ValueError('no checkpoint for resume')

        if self.loss_type == 'kd':
            if not classes_in_teacher == self.num_classes:
                self.validate('teacher', self.val_loader, model_teacher, criterion)
        
        logging.info('epoch, train accuracy, train loss, val accuracy, val loss')
        
        # train the model
        epoch = start_epoch
        while epoch < self.epochs:

            if self.loss_type == 'kd':
                train_obj, train_top1_acc, train_top5_acc = self.train_kd(epoch, self.train_loader, self.model_student, model_teacher, criterion_kd, optimizer, scheduler)
            elif self.loss_type == 'ce':
                train_obj, train_top1_acc, train_top5_acc = self.train(epoch, self.train_loader, self.model_student, criterion, optimizer, scheduler)
            elif self.loss_type == 'ls':
                train_obj, train_top1_acc, train_top5_acc = self.train(epoch, self.train_loader, self.model_student, criterion_smooth, optimizer, scheduler)
            else:
                raise ValueError('unsupport loss_type')

            valid_obj, valid_top1_acc, valid_top5_acc = self.validate(epoch, self.val_loader, self.model_student, criterion)
            
            logging.info("{}, {}, {}, {}, {}".format(epoch, train_top1_acc, train_obj, valid_top1_acc.item(), valid_obj))
            
            is_best = False
            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                is_best = True

            save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model_student.state_dict(),
                'best_top1_acc': best_top1_acc,
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                }, is_best, self.save)

            epoch += 1

        training_time = (time.time() - start_t) / 3600
        print('total training time = {} hours'.format(training_time))
        logging.info('total training time = {} hours'.format(training_time))
        print('* best acc = {}'.format(best_top1_acc))
        logging.info('* best acc = {}'.format(best_top1_acc))
        
    
    def train_kd(self, epoch, train_loader, model_student, model_teacher, criterion, optimizer, scheduler):
        
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        model_student.train()
        model_teacher.eval()
        end = time.time()
        scheduler.step()

        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('learning_rate:', cur_lr)

        for i, (images, target) in enumerate(train_loader):
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
            losses.update(loss.item(), n)   #accumulated loss
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
                adaptive_clip_grad(parameters_list, clip_factor=self.clip_value)

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i%50 == 0:
                progress.display(i)

        return losses.avg, top1.avg, top5.avg
    
    def train(self, epoch, train_loader, model_student, criterion, optimizer, scheduler):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        model_student.train()
        end = time.time()
        scheduler.step()

        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('learning_rate:', cur_lr)

        for i, (images, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            images = images.cuda()
            target = target.cuda()

            # compute outputy
            logits_student = model_student(images)
            loss = criterion(logits_student, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)   #accumulated loss
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
                adaptive_clip_grad(parameters_list, clip_factor=self.clip_value)

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i%50 == 0:
                progress.display(i)

        return losses.avg, top1.avg, top5.avg

    def validate(self, epoch, val_loader, model, criterion):
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

                if i%50 == 0:
                    progress.display(i)

            print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

        return losses.avg, top1.avg, top5.avg

