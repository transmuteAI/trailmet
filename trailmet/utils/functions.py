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
import random
import os
import numpy as np
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
import re
import math

__all__ = [
    'AverageMeter',
    'save_checkpoint',
    'accuracy',
    'seed_everything',
    'pdist',
    'CrossEntropyLabelSmooth',
    'adjust_learning_rate',
    'strlist_to_list',
    'get_optimizer',
    'lp_loss',
    'extract_sparsity',
    'chip_adjust_learning_rate',
]


def chip_adjust_learning_rate(self, optimizer, epoch, step, len_iter):
    if self.lr_type == 'step':
        factor = epoch // 125
        # if epoch >= 80:
        #     factor = factor + 1
        lr = self.learning_rate * (0.1**factor)

    elif self.lr_type == 'step_5':
        factor = epoch // 10
        if epoch >= 80:
            factor = factor + 1
        lr = self.learning_rate * (0.5**factor)

    elif self.lr_type == 'cos':  # cos without warm-up
        if self.epochs > 5:
            lr = (0.5 * self.learning_rate *
                  (1 + math.cos(math.pi * (epoch - 5) / (self.epochs - 5))))
        else:
            lr = self.learning_rate

    elif self.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = self.learning_rate * (decay**(epoch // step))

    elif self.lr_type == 'fixed':
        lr = self.learning_rate
    else:
        raise NotImplementedError

    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_iter) / (5.0 * len_iter)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if step == 0:
        print('learning_rate: ' + str(lr))


def extract_sparsity(sparsity):
    cprate_str = sparsity
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

    return cprate


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """Loss function measured in Lp Norm."""
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


def seed_everything(seed):
    'sets the random seed to ensure reproducibility'
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified
    values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value."""

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


def save_checkpoint(state, is_best, save, file_name=None):
    if not os.path.exists(save):
        os.makedirs(save)
    if file_name is not None:
        filename = os.path.join(save, f'{file_name}.pth.tar')
    else:
        filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) -
           2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


# label smooth
class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1),
                                                       1)
        targets = (1 -
                   self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def adjust_learning_rate(optimizer, epoch, num_epochs, scheduler_type, lr):
    """Sets the learning rate to the initial LR decayed by 2 every 30
    epochs."""
    if scheduler_type == 1:
        new_lr = lr * (0.5**(epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        if epoch in [num_epochs * 0.5, num_epochs * 0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1


def strlist_to_list(sstr, ttype):
    """
    example:
    # self.args.stage_pr = [0, 0.3, 0.3, 0.3, 0, ]
    # self.args.skip_layers = ['1.0', '2.0', '2.3', '3.0', '3.5', ]
    turn these into a list of <ttype> (float or str or int etc.)
    """
    if not sstr:
        return sstr
    out = []
    sstr = sstr.strip()
    if sstr.startswith('[') and sstr.endswith(']'):
        sstr = sstr[1:-1]
    for x in sstr.split(','):
        x = x.strip()
        if x:
            x = ttype(x)
            out.append(x)
    return out


def get_optimizer(self, optimizer_name: str, model, lr, weight_decay):
    """Returns the optimizer with the given name."""
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=lr,
                              weight_decay=weight_decay)
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=lr,
                               weight_decay=weight_decay)
    else:
        raise ValueError('Unknown optimizer: %s' % optimizer_name)
    return optimizer
