import random
import os
import numpy as np
import torch
import shutil
import torch.nn as nn
import torch.optim as optim


__all__ = ["AverageMeter", "save_checkpoint", "accuracy", "seed_everything", "pdist", "CrossEntropyLabelSmooth", "adjust_learning_rate", "strlist_to_list", "get_optimizer", "lp_loss"]

def lp_loss(pred, tgt, p=2.0, reduction='none'):
        """loss function measured in Lp Norm"""
        if reduction == 'none':
            return (pred-tgt).abs().pow(p).sum(1).mean()
        else:
            return (pred-tgt).abs().pow(p).mean()


def seed_everything(seed):
    "sets the random seed to ensure reproducibility"
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def save_checkpoint(state, is_best, save, file_name=None):
    if not os.path.exists(save):
        os.makedirs(save)
    if file_name is not None:
        filename = os.path.join(save, f"{file_name}.pth.tar")
    else:
        filename = os.path.join(save, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, "model_best.pth.tar")
        shutil.copyfile(filename, best_filename)


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

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
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

def adjust_learning_rate(optimizer, epoch, num_epochs, scheduler_type, lr):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    if scheduler_type==1:
        new_lr = lr * (0.5 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        if epoch in [num_epochs*0.5, num_epochs*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

def strlist_to_list(sstr, ttype):
    '''
        example:
        # self.args.stage_pr = [0, 0.3, 0.3, 0.3, 0, ]
        # self.args.skip_layers = ['1.0', '2.0', '2.3', '3.0', '3.5', ]
        turn these into a list of <ttype> (float or str or int etc.)
    '''
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
        """returns the optimizer with the given name"""
        if optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Unknown optimizer: %s' % optimizer_name)
        return optimizer
