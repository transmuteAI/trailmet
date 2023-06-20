import os
import argparse
import numpy as np
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
import sys
sys.argv=['']
del sys
  
def build_model(arch, pretrain_dir, gpu):
    model = eval(arch)(sparsity=[0.]*100).to(self.device)
    print('Loading Pretrained Model')
    if self.arch=='resnet_56':
        checkpoint = torch.load(pretrain_dir, map_location='cuda:'+gpu)
    else:
        checkpoint = torch.load(pretrain_dir)
    if self.arch=='resnet_50':
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['state_dict'])
            
class record_config():
    def __init__(self,result_dir):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        result_dir = Path(result_dir)

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(result_dir)

        config_dir = result_dir / 'config.txt'


    def get_logger(file_path):

        logger = logging.getLogger('gal')
        log_format = '%(asctime)s | %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

        return logger
            
#label smooth
class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, is_best, save):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def progress_bar(current, total, msg=None):
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)

    TOTAL_BAR_LENGTH = 65.
    last_time = time.time()
    begin_time = last_time

    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('utils')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f        
        
class Chip(BasePruning):
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
        self.conv_index = self.CFG['conv_index']
    
    
    def get_feature_hook(self,module, input, output):
        if not os.path.isdir('conv_feature_map/' + self.arch + '_repeat%d' % (self.repeat)):
            os.makedirs('conv_feature_map/' + self.arch + '_repeat%d' % (self.repeat))
        np.save('conv_feature_map/' + self.arch + '_repeat%d' % (self.repeat) + '/conv_feature_map_'+ str(self.conv_index) + '.npy',
                output.cpu().numpy())
        self.conv_index += 1
    
    def inference(self):
        model = self.model
        model.eval()
        repeat = self.repeat
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.dataloaders['train']):
                #use 5 batches to get feature maps.
                if batch_idx >= repeat:
                    break

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                model(inputs)
                
    
    def reduced_1_row_norm(self,input, row_index, data_index):
        input[data_index, row_index, :] = torch.zeros(input.shape[-1])
        m = torch.norm(input[data_index, :, :], p = 'nuc').item()
        return m
    
    def ci_score(self,path_conv):
        conv_output = torch.tensor(np.round(np.load(path_conv), 4))
        conv_reshape = conv_output.reshape(conv_output.shape[0], conv_output.shape[1], -1)

        r1_norm = torch.zeros([conv_reshape.shape[0], conv_reshape.shape[1]])
        for i in range(conv_reshape.shape[0]):
            for j in range(conv_reshape.shape[1]):
                r1_norm[i, j] = self.reduced_1_row_norm(conv_reshape.clone(), j, data_index = i)

        ci = np.zeros_like(r1_norm)

        for i in range(r1_norm.shape[0]):
            original_norm = torch.norm(torch.tensor(conv_reshape[i, :, :]), p='nuc').item()
            ci[i] = original_norm - r1_norm[i]

        # return shape: [batch_size, filter_number]
        return ci
    
    def mean_repeat_ci(self,repeat, num_layers):
        layer_ci_mean_total = []
        for j in range(num_layers):
            repeat_ci_mean = []
            for i in range(repeat):
                index = j * repeat + i + 1
                # add
                path_conv = "./conv_feature_map/{0}_repeat5/conv_feature_map_tensor({1}).npy".format(str(self.arch), str(index))
                
                batch_ci = self.ci_score(path_conv)
                single_repeat_ci_mean = np.mean(batch_ci, axis=0)
                repeat_ci_mean.append(single_repeat_ci_mean)

            layer_ci_mean = np.mean(repeat_ci_mean, axis=0)
            layer_ci_mean_total.append(layer_ci_mean)

        return np.array(layer_ci_mean_total)
    
    def load_resnet_model(self,model, oristate_dict):
        record_config(self.result_dir)
        logger = record_config.get_logger(os.path.join(self.result_dir, 'logger.log'))
        if len(self.gpu)>1:
            name_base='module.'
        else:
            name_base=''
        cfg = {'resnet_50': [3, 4, 6, 3],}

        state_dict = model.state_dict()

        current_cfg = cfg[self.arch]
        last_select_index = None

        all_honey_conv_weight = []

        bn_part_name=['.weight','.bias','.running_mean','.running_var']#,'.num_batches_tracked']
        prefix = self.ci_dir+'/ci_conv'
        subfix = ".npy"
        cnt=1

        conv_weight_name = 'conv1.weight'
        all_honey_conv_weight.append(conv_weight_name)
        oriweight = oristate_dict[conv_weight_name]
        curweight = state_dict[name_base+conv_weight_name]
        orifilter_num = oriweight.size(0)
        currentfilter_num = curweight.size(0)

        if orifilter_num != currentfilter_num:
            logger.info('loading ci from: ' + prefix + str(cnt) + subfix)
            ci = np.load(prefix + str(cnt) + subfix)
            select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
            select_index.sort()

            for index_i, i in enumerate(select_index):
                state_dict[name_base+conv_weight_name][index_i] = \
                    oristate_dict[conv_weight_name][i]
                for bn_part in bn_part_name:
                    state_dict[name_base + 'bn1' + bn_part][index_i] = \
                        oristate_dict['bn1' + bn_part][i]

            last_select_index = select_index
        else:
            state_dict[name_base + conv_weight_name] = oriweight
            for bn_part in bn_part_name:
                state_dict[name_base + 'bn1' + bn_part] = oristate_dict['bn1'+bn_part]

        state_dict[name_base + 'bn1' + '.num_batches_tracked'] = oristate_dict['bn1' + '.num_batches_tracked']

        cnt+=1
        for layer, num in enumerate(current_cfg):
            layer_name = 'layer' + str(layer + 1) + '.'

            for k in range(num):
                iter = 3
                if k==0:
                    iter +=1
                for l in range(iter):
                    record_last=True
                    if k==0 and l==2:
                        conv_name = layer_name + str(k) + '.downsample.0'
                        bn_name = layer_name + str(k) + '.downsample.1'
                        record_last=False
                    elif k==0 and l==3:
                        conv_name = layer_name + str(k) + '.conv' + str(l)
                        bn_name = layer_name + str(k) + '.bn' + str(l)
                    else:
                        conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                        bn_name = layer_name + str(k) + '.bn' + str(l + 1)

                    conv_weight_name = conv_name + '.weight'
                    all_honey_conv_weight.append(conv_weight_name)
                    oriweight = oristate_dict[conv_weight_name]
                    curweight = state_dict[name_base+conv_weight_name]
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)

                    if orifilter_num != currentfilter_num:
                        logger.info('loading ci from: ' + prefix + str(cnt) + subfix)
                        ci = np.load(prefix + str(cnt) + subfix)
                        select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                        select_index.sort()

                        if last_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_select_index):
                                    state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                        oristate_dict[conv_weight_name][i][j]


                                for bn_part in bn_part_name:
                                    state_dict[name_base + bn_name + bn_part][index_i] = \
                                        oristate_dict[bn_name + bn_part][i]

                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base+conv_weight_name][index_i] = \
                                    oristate_dict[conv_weight_name][i]

                                for bn_part in bn_part_name:
                                    state_dict[name_base + bn_name + bn_part][index_i] = \
                                        oristate_dict[bn_name + bn_part][i]

                        if record_last:
                            last_select_index = select_index

                    elif last_select_index is not None:
                        for index_i in range(orifilter_num):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][index_i][j]

                        for bn_part in bn_part_name:
                            state_dict[name_base + bn_name + bn_part] = \
                                oristate_dict[bn_name + bn_part]

                        if record_last:
                            last_select_index = None

                    else:
                        state_dict[name_base+conv_weight_name] = oriweight
                        for bn_part in bn_part_name:
                            state_dict[name_base + bn_name + bn_part] = \
                                oristate_dict[bn_name + bn_part]
                        if record_last:
                            last_select_index = None

                    state_dict[name_base + bn_name + '.num_batches_tracked'] = oristate_dict[bn_name + '.num_batches_tracked']
                    cnt+=1

        for name, module in model.named_modules():
            name = name.replace('module.', '')
            if isinstance(module, nn.Conv2d):
                conv_name = name + '.weight'
                if conv_name not in all_honey_conv_weight:
                    state_dict[name_base+conv_name] = oristate_dict[conv_name]

            elif isinstance(module, nn.Linear):
                state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

        model.load_state_dict(state_dict)
    
            
    def adjust_learning_rate(self,optimizer, epoch, step, len_iter):
        
        record_config(self.result_dir)
        logger = record_config.get_logger(os.path.join(self.result_dir, 'logger.log'))
        
        if self.lr_type == 'step':
            factor = epoch // 125
        # if epoch >= 80:
        #     factor = factor + 1
            lr = self.learning_rate * (0.1 ** factor)

        elif self.lr_type == 'step_5':
            factor = epoch // 10
            if epoch >= 80:
                factor = factor + 1
            lr = self.learning_rate * (0.5 ** factor)

        elif self.lr_type == 'cos':  # cos without warm-up
            if self.epochs > 5:
                lr = 0.5 * self.learning_rate * (1 + math.cos(math.pi * (epoch - 5) / (self.epochs - 5)))
            else:
                lr = self.learning_rate

        elif self.lr_type == 'exp':
            step = 1
            decay = 0.96
            lr = self.learning_rate * (decay ** (epoch // step))

        elif self.lr_type == 'fixed':
            lr = self.learning_rate
        else:
            raise NotImplementedError

        if epoch < 5:
            lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if step == 0:
            logger.info('learning_rate: ' + str(lr))


    def compress_model(self): 
        model = self.model
        cov_layer = eval('model.maxpool')
        handler = cov_layer.register_forward_hook(self.get_feature_hook)
        self.inference()
        handler.remove()

        # ResNet50 per bottleneck
        for i in range(4):
            block = eval('model.layer%d' % (i + 1))
            for j in range(model.num_blocks[i]):
                cov_layer = block[j].relu1
                handler = cov_layer.register_forward_hook(self.get_feature_hook)
                self.inference()
                handler.remove()

                cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(self.get_feature_hook)
                self.inference()
                handler.remove()

                cov_layer = block[j].relu3
                handler = cov_layer.register_forward_hook(self.get_feature_hook)
                self.inference()
                handler.remove()

                if j==0:
                    cov_layer = block[j].relu3
                    handler = cov_layer.register_forward_hook(self.get_feature_hook)
                    self.inference()
                    handler.remove()


        
        repeat = self.repeat
        num_layers = self.num_layers
        save_path = 'CI_' + self.arch
        ci = self.mean_repeat_ci(repeat, num_layers)
        if self.arch == 'resnet_50':
            num_layers = 53
        for i in range(num_layers):
            print(i)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            np.save(save_path + "/ci_conv{0}.npy".format(str(i + 1)), ci[i])
        
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)

        #save old training file
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        cp_file_dir = os.path.join(self.result_dir, 'cp_file/' + now)
        if os.path.exists(self.result_dir+'/model_best.pth.tar'):
            if not os.path.isdir(cp_file_dir):
                os.makedirs(cp_file_dir)
            shutil.copy(self.result_dir+'/config.txt', cp_file_dir)
            shutil.copy(self.result_dir+'/logger.log', cp_file_dir)
            shutil.copy(self.result_dir+'/model_best.pth.tar', cp_file_dir)
            shutil.copy(self.result_dir + '/checkpoint.pth.tar', cp_file_dir)

        record_config(self.result_dir)
        logger = record_config.get_logger(os.path.join(self.result_dir, 'logger.log'))
        
        cudnn.benchmark = True
        cudnn.enabled=True
        logger.info("args = %s", self)

        if self.sparsity:
            import re
            cprate_str = self.sparsity
            cprate_str_list = cprate_str.split('+')
            pat_cprate = re.compile(r'\d+\.\d*')
            pat_num = re.compile(r'\*\d+')
            cprate = []
            for x in cprate_str_list:
                num = 1
                find_num = re.findall(pat_num, x)
                if find_num:
                    assert len(find_num) == 1
                    num = int(find_num[0].replace('*', ''))
                find_cprate = re.findall(pat_cprate, x)
                assert len(find_cprate) == 1
                cprate += [float(find_cprate[0])] * num

            sparsity = cprate

        # load model
        logger.info('sparsity:' + str(sparsity))
        logger.info('==> Building model..')
        model = eval(self.arch)(sparsity=sparsity).cuda()
        logger.info(model)


        CLASSES = 1000
        print_freq = 128000//self.batch_size
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        criterion_smooth = CrossEntropyLabelSmooth(CLASSES, self.label_smooth)
        criterion_smooth = criterion_smooth.cuda()


        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        lr_decay_step = list(map(int, self.lr_decay_step.split(',')))
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

        start_epoch = 0
        best_top1_acc= 0
        best_top5_acc= 0

        # load the checkpoint if it exists
        checkpoint_dir = os.path.join(self.result_dir, 'checkpoint.pth.tar')

        logger.info('resuming from pretrain model')
        origin_model = eval(self.arch)(sparsity=[0.] * 100).cuda()
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
        while epoch < self.epochs:

            train_obj, train_top1_acc,  train_top5_acc = self.train(epoch,  self.dataloaders['train'], model, criterion_smooth, optimizer, scaler)
            valid_obj, valid_top1_acc, valid_top5_acc = self.validate(epoch, self.dataloaders['val'], model, criterion)

            is_best = False
            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                best_top5_acc = valid_top5_acc
                is_best = True

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_top1_acc': best_top1_acc,
                'best_top5_acc': best_top5_acc,
                'optimizer' : optimizer.state_dict(),
                }, is_best, self.result_dir)

            epoch += 1
            logger.info("=>Best accuracy Top1: {:.3f}, Top5: {:.3f}".format(best_top1_acc, best_top5_acc))

        training_time = (time.time() - start_t) / 36000
        logger.info('total training time = {} hours'.format(training_time))
    
    def train(self,epoch, train_loader, model, criterion, optimizer, scaler = None):
        record_config(self.result_dir)
        logger = record_config.get_logger(os.path.join(self.result_dir, 'logger.log'))
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        model.train()
        end = time.time()
        #scheduler.step()

        num_iter = len(train_loader)

        print_freq = num_iter // 10

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.cuda()
            targets = targets.cuda()
            data_time.update(time.time() - end)

            self.adjust_learning_rate(optimizer, epoch, batch_idx, num_iter)

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

            if batch_idx % print_freq == 0 and batch_idx != 0:
                logger.info(
                    'Epoch[{0}]({1}/{2}): '
                    'Loss {loss.avg:.4f} '
                    'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                        epoch, batch_idx, num_iter, loss=losses,
                        top1=top1, top5=top5))
        return losses.avg, top1.avg, top5.avg

    def validate(self, epoch, val_loader, model, criterion):
        record_config(self.result_dir)
        logger = record_config.get_logger(os.path.join(self.result_dir, 'logger.log'))
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        model.eval()
        with torch.no_grad():
            end = time.time()
            for batch_idx, (images, targets) in enumerate(val_loader):
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

            logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                        .format(top1=top1, top5=top5))

        return losses.avg, top1.avg, top5.avg

