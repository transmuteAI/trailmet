from __future__ import print_function
""
import sys
sys.path.append("../../../../../")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
#from ...models import models
from torch.autograd import Variable

import shutil
import numpy as np
from trailmet.models import channel_selection
from trailmet.algorithms import BaseAlgorithm
import trailmet.models as models
torch.cuda.manual_seed(42)
"""
making train loaders and test loaders from given dataset
"""
""
class Process(BaseAlgorithm):
        def __init__(self,params):
            super(Process,self).__init__()
            args = params
            kwargs = {'num_workers': 2, 'pin_memory': True}
            if(args.data=='CIFAR10'):
                train_loader = torch.utils.data.DataLoader(
                    datasets.CIFAR10('./data', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.Pad(4),
                                      transforms.RandomCrop(32),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ])),
                batch_size=args.train_bs, shuffle=True, **kwargs)
                test_loader = torch.utils.data.DataLoader(
                    datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
                            batch_size=args.test_bs, shuffle=True, **kwargs)
            else:
              assert(args.train_loader is not None and args.test_loader is not None)
              train_loader = args.train_loader 
              test_loader = args.test_loader

            ########################################################################
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


            """the updated count of number of channels after pruning"""
            # if(args.fine_tune):
            #     from prunit import cfg
            #     print(cfg)
            #cfg=[6, 14, 8, 9, 6, 6, 8, 6, 4, 10, 14, 10, 5, 9, 4, 11, 9, 11, 7, 6, 8, 12, 11, 11, 11, 14, 6, 13, 16, 12, 4, 7, 5, 5, 7, 9, 13, 15, 14, 5, 9, 11, 6, 4, 8, 2, 2, 4, 5, 7, 10, 11, 11, 14, 35, 31, 31, 18, 18, 29, 23, 22, 30, 20, 24, 32, 11, 14, 31, 19, 24, 30, 22, 22, 29, 21, 27, 30, 20, 18, 29, 20, 21, 26, 17, 23, 29, 15, 17, 29, 27, 22, 28, 25, 24, 31, 25, 23, 28, 24, 23, 28, 19, 15, 30, 13, 15, 22, 108, 64, 64, 24, 40, 64, 26, 39, 64, 32, 53, 60, 38, 51, 62, 45, 60, 64, 53, 61, 64, 53, 59, 61, 58, 64, 63, 71, 62, 60, 55, 58, 54, 80, 62, 60, 59, 56, 53, 76, 61, 50, 73, 59, 50, 60, 51, 45, 69, 51, 42, 45, 38, 38, 68]


            """loading the models to gpu"""
            #########################################################################
            if args.fine_tune:
                checkpoint = torch.load(args.path)
                cfg = checkpoint['cfg']
                if (args.arch == 'vgg'):
                    model = models.vgg(cfg=cfg , num_classes = args.num_classes )
                    model = model.to(device)
                elif(args.arch== 'resnet'):
                    model = models.__dict__[args.arch](dataset=args.data, depth=args.depth, cfg=cfg , num_classes = args.num_classes)
                    model = model.to(device)
                #checkpoint = torch.load(args.path)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                if (args.arch == 'vgg'):
                    model = models.vgg()
                    model = model.to(device)
                elif(args.arch=='resnet'):
                    model = models.__dict__[args.arch](dataset=args.data, depth=args.depth)
                    model = model.to(device)
            print(model)
            optimizer = self.get_optimizer(optimizer_name=args.optimizer_name , model=model, lr=args.lr , weight_decay = args.weight_decay) if args.optimizer_name is not None else optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            criteria= nn.CrossEntropyLoss()
            ############################################################################


            """resume training of saved checkpoint"""
            ############################################################################

            if args.resume:
                if os.path.isfile(args.resume):
                    model.to(device)
                    print("=> loading checkpoint '{}'".format(args.resume))
                    checkpoint = torch.load(args.resume)
                    args.start_epoch = checkpoint['epoch']
                    best_prec1 = checkpoint['best_prec1']
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                          .format(args.resume, checkpoint['epoch'], best_prec1))
                else:
                    print("=> no checkpoint found at '{}'".format(args.resume))

            ############################################################################

            """separate function to update the values of scaling parameter"""
            def updateBN(model= model):
                
              for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                  m.weight.grad.data.add_(args.thr*torch.sign(m.weight.data))
                

            ###########################################################################

            """training function"""
            # def train(epoch):
            #   model.train()
            #   for batch_idx, (data, target) in enumerate(train_loader):
            #     data=data.to(device)
            #     target=target.to(device)
            #     optimizer.zero_grad()
            #     output = model(data)
            #     loss = criteria(output, target)
            #     loss.backward()
            #     if args.sparsity_reg:  # here, we are updating the values of scaling parameter
            #       updateBN()
            #     optimizer.step()
            #     if (batch_idx % args.log_interval) == 0:
            #       print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss))

            ###############################################################################


            """testing function"""
            ###############################################################################
            # def test():
            #   model.eval()
            #   test_loss = 0
            #   correct = 0
            #   with torch.no_grad():
            #     for data, target in test_loader:
            #       data=data.to(device)
            #       target=target.to(device)
            #       output = model(data)
            #       test_loss += criteria(output, target).item()
            #       _,pred = torch.max(output,1)
            #       correct+=(pred == target).sum().item()

            #   test_loss /= len(test_loader.dataset)
            #   print('\nTest set: Average loss: {}, Accuracy: {}/{} ({:.1f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
            #   return correct / float(len(test_loader.dataset))

            ################################################################################

            """save the details of current checkpoint and also saves information of best model """

            def save_checkpoint(state, is_best, filename=args.file_name):
                torch.save(state, filename)
                if is_best:
                    shutil.copyfile(filename, 'model_best.pth.tar')

            #################################################################################

            """putting it all together """
            best_prec1 = 0.
            for epoch in range(0, args.epochs):
              print(epoch)
              if epoch in [args.epochs*0.5, args.epochs*0.75]:
                for param_group in optimizer.param_groups:
                  param_group['lr'] *= 0.1
              loss = self.train_one_epoch(model=model , dataloader=train_loader , loss_fn=criteria , optimizer=optimizer , extra_functionality=None)
              updateBN()
              prec1, loss_test = self.test(model = model , dataloader = test_loader , loss_fn = criteria)
              print("Train set :: Average loss: {} \n".format(loss))
              print('\nTest set: Average loss: {}, Accuracy: {} \n'.format(loss_test, prec1*100 , " "))
#             
              is_best = prec1>best_prec1
              save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, is_best)
            model.load_state_dict(torch.load(args.file_name)['state_dict'])
            return model

#from ...models import models
""
# class get_params():
#   def __init__(self,dataset,test_bs,percent,model,save,depth,arch):
#     self.dataset=dataset
#     self.test_bs=test_bs
#     self.percent=percent  # percentage of network to be pruned
#     self.model=model      # path where the model to be pruned is saved
#     self.save=save        # path where pruned model will be saved
#     self.depth=depth      # depth of model if arch is resnet
#     self.arch=arch        # vgg-16 and resnet family is supported
# initial=get_params('cifar10',64,0.6,'/content/drive/MyDrive/VGG-16 NS/CIFAR-10/trained/resnet_model_best.pth.tar','/content/drive/MyDrive/VGG-16 NS/CIFAR-10/pruned/resnet_pruned1.pth.tar',164,'resnet')
""
"""
defining model architecture
"""
class prune_it:
    def __init__(self,params):
                args = params
                if (args.arch == 'vgg'):
                    model = models.vgg()
                elif (args.arch == 'resnet'):
                    model = models.__dict__[args.arch](dataset=args.data, depth=args.depth)
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

#                 def test():
#                     kwargs = {'num_workers': 2, 'pin_memory': True}
#                     test_loader = torch.utils.data.DataLoader(
#                         datasets.CIFAR10('./data', train=False,download=True, transform=transforms.Compose([
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
#                         batch_size=args.test_bs, shuffle=True, **kwargs)
#                     model.eval()
#                     correct = 0
#                     for data, target in test_loader:
#                         data, target = data.cuda(), target.cuda()
#                         data, target = Variable(data, volatile=True), Variable(target)
#                         output = model(data)
#                         pred = output.data.max(1, keepdim=True)[1]
#                         correct += pred.eq(target.data.view_as(pred)).cpu().sum()

#                     print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
#                     return correct / float(len(test_loader.dataset))

                ############################################################################
                """defining a pruned model"""
                print(cfg)
                if (args.arch == 'vgg'):
                    newmodel = models.vgg()
                elif (args.arch == 'resnet'):
                    newmodel = models.__dict__[args.arch](dataset=args.data, depth=args.depth)
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
                return model
            ##################################################################################
class OfNoUse:
  pass
class NetworkSlimming:
  def __init__(self ,data_loaders, path , save , model,   **kwargs):
    #super(NetworkSlimming,self).__init__(**kwargs)
    self.args = OfNoUse()
    self.kwargs = kwargs
    self.args.data = self.kwargs['NETWORK_SLIMMING'].get('DATA' ,None)
    self.args.num_classes = self.kwargs['NETWORK_SLIMMING'].get('NUM_CLASSES' , 10)      # dataset on which model is trained
    self.args.sparsity_reg = self.kwargs['NETWORK_SLIMMING'].get('SPARSITY_REG' , True)    # true if training is done with sparsity regularization
    self.args.thr = self.kwargs['NETWORK_SLIMMING'].get('THR' ,1e-5)      # the sparsity regularization hyperparameter value
    self.args.train_loader = data_loaders['train']
    self.args.test_loader = data_loaders['test']
    self.args.fine_tune = self.kwargs['NETWORK_SLIMMING'].get('FINE_TUNE' ,False)        # true if pruned model is being fine-tuned
    self.args.path = self.kwargs['NETWORK_SLIMMING'].get('PATH' ,None)      # path from where the pruned model is loaded
    self.args.resume = self.kwargs['NETWORK_SLIMMING'].get('RESUME' ,False)      # true of we have to resume training of some model whose checkpoint is saved
    self.args.train_bs = self.kwargs['NETWORK_SLIMMING'].get('TRAIN_BS' ,64)      # training batch size
    self.args.test_bs = self.kwargs['NETWORK_SLIMMING'].get('TEST_BS' ,256)        # test batch size
    self.args.epochs = self.kwargs['NETWORK_SLIMMING'].get('EPOCHS' ,100)
    self.args.optimizer_name = self.kwargs['NETWORK_SLIMMING'].get('OPTIMIZER_NAME', None)
    self.args.lr = self.kwargs['NETWORK_SLIMMING'].get('LR' ,1e-1)
    self.args.momentum = self.kwargs['NETWORK_SLIMMING'].get('MOMENTUM' ,0.9)
    self.args.weight_decay = self.kwargs['NETWORK_SLIMMING'].get('WEIGHT_DECAY' ,1e-4)
    self.args.log_interval = self.kwargs['NETWORK_SLIMMING'].get('LOG_INTERVAL' , 100)     # number of intervals after which accuracy and loss values are printed during training
    self.args.arch = self.kwargs['NETWORK_SLIMMING'].get('ARCH' ,'vgg')      # model architecture
    self.args.depth = self.kwargs['NETWORK_SLIMMING'].get('DEPTH' , 164) 
    self.args.percent = self.kwargs['NETWORK_SLIMMING'].get('PERCENT' , 0.6)
    self.args.path = path
    self.args.save = save
    self.args.model = model
           # depth of model (if resnet is being used)
    if(self.args.fine_tune):
      assert(self.args.path is not None)
      assert(self.args.save is not None)
      self.args.file_name = './pruned_model_best.pth.tar'
    else:
      self.args.save = './pruned_checkpoint.pth.tar'
      self.args.file_name = f'./{self.args.arch}_checkpoint.pth.tar'
      self.args.model = 'model_best.pth.tar' 
      self.args.path = './pruned_checkpoint.pth.tar'



  def base_line(self):
    assert(self.args.fine_tune == False)
    x1 = Process(params=self.args)
    return x1

  def prune(self):
    assert(self.args.save is not None)
    assert(self.args.model is not None)
    x1 = prune_it(params=self.args)
    return x1

  def fine_tune(self):
    assert(self.args.fine_tune == True and self.args.sparsity_reg == False)
    x1 = Process(params=self.args)
    return x1

  def do_all(self):
    assert(self.args.fine_tune == False)
    m1 = self.base_line()
    m1 = self.prune()
    self.args.sparsity_reg = False
    self.args.fine_tune = True
    return self.fine_tune()