#######################################################################
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets,transforms

import numpy as np
import models
from models import channel_selection
########################################################################

class get_params():
  def __init__(self,dataset,test_bs,percent,model,save,depth,arch):
    self.dataset=dataset
    self.test_bs=test_bs
    self.percent=percent  # percentage of network to be pruned
    self.model=model      # path where the model to be pruned is saved
    self.save=save        # path where pruned model will be saved
    self.depth=depth      # depth of model if arch is resnet
    self.arch=arch        # vgg-16 and resnet family is supported

args=get_params('cifar10',64,0.6,'/content/drive/MyDrive/VGG-16 NS/CIFAR-10/trained/resnet_model_best.pth.tar','/content/drive/MyDrive/VGG-16 NS/CIFAR-10/pruned/resnet_pruned1.pth.tar',164,'resnet')

##########################################################################
"""defining model architecture"""

if (args.arch == 'vgg'):
    model = models.vgg()
elif (args.arch == 'resnet'):
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=model.to(device)

############################################################################
"""loading model from path"""

if os.path.isfile(args.model):
  print("=> loading checkpoint '{}'".format(args.model))
  checkpoint = torch.load(args.model)
  args.start_epoch = checkpoint['epoch']
  best_prec1 = checkpoint['best_prec1']
  model.load_state_dict(checkpoint['state_dict'])
  print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(args.model, checkpoint['epoch'], best_prec1))

############################################################################

total = 0  # total number of scaling parameters
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

bn = torch.zeros(total)   # for saving all scaling parameters
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre = y[thre_index]   # the scaling threshold value, if a channel has scaling value lesser than threshold, it will be pruned

##############################################################################

pruned = 0     # total number of channels that are pruned
cfg = []       # list of number of channels that will be left after pruning
cfg_mask = []  # list that contains mask, in each mask we have information
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.clone()
        mask = weight_copy.abs().gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned/total

##############################################################################

def test():
    kwargs = {'num_workers': 2, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False,download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.test_bs, shuffle=True, **kwargs)
    model.eval()
    correct = 0
    for data, target in test_loader:
      data, target = data.cuda(), target.cuda()
      data, target = Variable(data, volatile=True), Variable(target)
      output = model(data)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

############################################################################
"""defining a pruned model"""
print(cfg)
if (args.arch == 'vgg'):
    newmodel = models.vgg()
elif (args.arch == 'resnet'):
    newmodel = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
newmodel=newmodel.to(device)
print(len(cfg))

############################################################################
"""copying the weights corresponding to channels left after pruning in the new model"""

if(args.arch=="vgg"):
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            m1.weight.data = m0.weight.data[idx1].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            m1.running_mean = m0.running_mean[idx1].clone()
            m1.running_var = m0.running_var[idx1].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))   # indexes of channels that are left in previous layer's ouput
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))     # indexes of filters that are selected corresponding to channels left after applying pruning
            print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
            w = m0.weight.data[:, idx0, :, :].clone()
            w = w[idx1, :, :, :].clone()
            m1.weight.data = w.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            m1.weight.data = m0.weight.data[:, idx0].clone()

elif(args.arch=='resnet'):
    old_modules = list(model.modules())
    new_modules = list(newmodel.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0

    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if isinstance(old_modules[layer_id + 1], channel_selection):
                # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
                # This convers the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

                # If the current convolution is not the last convolution in the residual block, then we can change the
                # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
                if conv_count % 3 != 1:
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                continue

            # We need to consider the case where there are downsampling convolutions.
            # For these convolutions, we just copy the weights.
            m1.weight.data = m0.weight.data.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

##################################################################################

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, args.save)

##################################################################################
print(newmodel)
model = newmodel
test()

