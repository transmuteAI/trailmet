
import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm import tqdm_notebook
from ..algorithms import BaseAlgorithm

class BaseQuantization(BaseAlgorithm):
    """base class for quantization algorithms"""
    def __init__(self, **kwargs):
        super(BaseQuantization, self).__init__(**kwargs)
        pass

    def quantize(self, model, dataloaders, method, **kwargs):
        pass
    
    # To do : refactor changes for round_ste -> class RoundSTE modification
    def round_ste(x: torch.Tensor):
        """
        Implement Straight-Through Estimator for rounding operation.
        """
        return (x.round() - x).detach() + x

    def get_calib_samples(self, train_loader, num_samples):
        """
        Get calibration-set samples for finetuning weights and clipping parameters
        """
        calib_data = []
        for batch in train_loader:
            calib_data.append(batch[0])
            if len(calib_data)*batch[0].size(0) >= num_samples:
                break
        return torch.cat(calib_data, dim=0)[:num_samples]

# the lp_loss, accuracy and test functions can be shifted to the BaseAlogorithm class in ./algorithms.py
    
    def lp_loss(pred, tgt, p=2.0, reduction='none'):
        """loss function measured in Lp Norm"""
        if reduction == 'none':
            return (pred-tgt).abs().pow(p).sum(1).mean()
        else:
            return (pred-tgt).abs().pow(p).mean()

    def accuracy(self, output, target, topk=(1,)):
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

    def test(self, model, dataloader, loss_fn=None, device=None):
        """This method is used to test the performance of the trained model."""
        if device is None:
            device = next(model.parameters()).device()
        else:
            model.to(device)
        model.eval()
        counter=0
        tk1=tqdm_notebook(dataloader, total=len(dataloader))
        running_acc1=0
        running_acc5=0
        running_loss=0
        with torch.no_grad():
            for images, targets in tk1:
                counter+=1
                images = images.to(device)
                targets = targets.to(device)
                if len(images)!=64:                 # To do: fix this
                    continue
                outputs = model(images)
                acc1, acc5 = self.accuracy(outputs, targets, topk=(1,5))
                running_acc1+=acc1[0].item()
                running_acc5+=acc5[0].item()
                if loss_fn is not None:
                    loss = loss_fn(outputs, targets)
                    running_loss+=loss.item()
                    tk1.set_postfix(loss=running_loss/counter, acc1=running_acc1/counter, acc5=running_acc5/counter)
                else:
                    tk1.set_postfix(acc1=running_acc1/counter, acc5=running_acc5/counter)
        return running_acc1/counter, running_acc5/counter, running_loss/counter


class RoundSTE(torch.autograd.Function):
    """customized round function with enabled backpropagation"""
    @staticmethod
    def forward(ctx, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class StraightThrough(nn.Module):
    """used to place an identity function in place of a non-differentail operator for gradient calculation"""
    def __int__(self):
        super().__init__()
        pass

    def forward(self, input):
        return input
    
class FoldBN():
    """used to help fold batch norm to prev layer activations"""
    def __init__(self):
        pass

    def _fold_bn(self, conv_module, bn_module):
        w = conv_module.weight.data
        y_mean = bn_module.running_mean
        y_var = bn_module.running_var
        safe_std = torch.sqrt(y_var + bn_module.eps)
        w_view = (conv_module.out_channels, 1, 1, 1)
        if bn_module.affine:
            weight = w * (bn_module.weight / safe_std).view(w_view)
            beta = bn_module.bias - bn_module.weight * y_mean / safe_std
            if conv_module.bias is not None:
                bias = bn_module.weight * conv_module.bias / safe_std + beta
            else:
                bias = beta
        else:
            weight = w / safe_std.view(w_view)
            beta = -y_mean / safe_std
            if conv_module.bias is not None:
                bias = conv_module.bias / safe_std + beta
            else:
                bias = beta
        return weight, bias


    def fold_bn_into_conv(self, conv_module, bn_module):
        w, b = self._fold_bn(conv_module, bn_module)
        if conv_module.bias is None:
            conv_module.bias = nn.Parameter(b)
        else:
            conv_module.bias.data = b
        conv_module.weight.data = w
        # set bn running stats
        bn_module.running_mean = bn_module.bias.data
        bn_module.running_var = bn_module.weight.data ** 2


    def reset_bn(self, module: nn.BatchNorm2d):
        if module.track_running_stats:
            module.running_mean.zero_()
            module.running_var.fill_(1-module.eps)
            # we do not reset number of tracked batches here
            # self.num_batches_tracked.zero_()
        if module.affine:
            init.ones_(module.weight)
            init.zeros_(module.bias)


    def is_bn(self, m):
        return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


    def is_absorbing(self, m):
        return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)


    def search_fold_and_remove_bn(self, model: nn.Module):
        model.eval()
        prev = None
        for n, m in model.named_children():
            if self.is_bn(m) and self.is_absorbing(prev):
                self.fold_bn_into_conv(prev, m)
                # set the bn module to straight through
                setattr(model, n, StraightThrough())
            elif self.is_absorbing(m):
                prev = m
            else:
                prev = self.search_fold_and_remove_bn(m)
        return prev


    def search_fold_and_reset_bn(self, model: nn.Module):
        model.eval()
        prev = None
        for n, m in model.named_children():
            if self.is_bn(m) and self.is_absorbing(prev):
                self.fold_bn_into_conv(prev, m)
                # reset_bn(m)
            else:
                self.search_fold_and_reset_bn(m)
            prev = m
