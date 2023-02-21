from __future__ import print_function
""
import sys
sys.path.append("../../../../")
import os
import torch
import torch.nn as nn
import torch.optim as optim
#from torchvision import datasets, transforms
#from ...models import models
from torch.autograd import Variable

import shutil
import numpy as np
from trailmet.models import channel_selection
from trailmet.algorithms import BaseAlgorithm
import trailmet.models as models
torch.cuda.manual_seed(42)

class Process(BaseAlgorithm):

    def __init__(self,params):
        super(Process,self).__init__()
        self.args = params
        kwargs = {'num_workers': 2, 'pin_memory': True}
        assert(self.args.train_loader is not None and self.args.test_loader is not None)
        self.train_loader = self.args.train_loader 
        self.test_loader = self.args.test_loader

        ########################################################################
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def create_model(self,device):
        #########################################################################
        if self.args.fine_tune:
            checkpoint = torch.load(self.args.path)
            cfg = checkpoint['cfg']
            if (self.args.arch == 'vgg'):
                model = models.vgg(cfg=cfg , num_classes = self.args.num_classes )
                model = model.to(device)
            elif(self.args.arch== 'resnet'):
                #model = models.__dict__[self.args.arch](dataset=self.args.data, depth=self.args.depth, cfg=cfg , num_classes = self.args.num_classes)
                model = models.make_ns_resnet(num_classes=self.args.num_classes , cfg = cfg)
                model = model.to(device) 
            #checkpoint = torch.load(self.args.path)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            if (self.args.arch == 'vgg'):
                model = models.vgg()
                model = model.to(device)
            elif(self.args.arch=='resnet'):
                #model = models.__dict__[self.args.arch](dataset=self.args.data, depth=self.args.depth)
                model = models.make_ns_resnet(num_classes=self.args.num_classes)
                model = model.to(device)
        self.model = model
        print(model)
        optimizer = self.get_optimizer(optimizer_name=self.args.optimizer_name , model=model, lr=self.args.lr , weight_decay = self.args.weight_decay) if self.args.optimizer_name is not None else optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        criteria= nn.CrossEntropyLoss()
        return optimizer,criteria

    def resume(self,device,model,optimizer):
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                model.to(device)
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                      .format(self.args.resume, checkpoint['epoch'], best_prec1))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        return model
    def updateBN(self,model):
          
          for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
              m.weight.grad.data.add_(self.args.thr*torch.sign(m.weight.data))
          return model


    def save_checkpoint(self,state, is_best, filename):
            torch.save(state, filename)
            if is_best:
                shutil.copyfile(filename, 'model_best.pth.tar')


    def train(self,model,optimizer,criteria,train_loader,test_loader):
        best_prec1 = 0.
        
        for epoch in range(0, self.args.epochs):
          print(epoch)
          if epoch in [self.args.epochs*0.5, self.args.epochs*0.75]:
            for param_group in optimizer.param_groups:
              param_group['lr'] *= 0.1
          loss = self.train_one_epoch(model=model , dataloader=train_loader , loss_fn=criteria , optimizer=optimizer , extra_functionality= None)
          for m in model.modules():
             if isinstance(m, nn.BatchNorm2d):
               m.weight.grad.data.add_(self.args.thr*torch.sign(m.weight.data))
          prec1, loss_test = self.test(model = model , dataloader = test_loader , loss_fn = criteria)
          print("Train set :: Average loss: {} \n".format(loss))
          print('\nTest set: Average loss: {}, Accuracy: {} \n'.format(loss_test, prec1*100 , " "))
        
          is_best = prec1>best_prec1
          self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best , filename= self.args.file_name)
        model.load_state_dict(torch.load(self.args.file_name)['state_dict'])
        return model
        
    def train_model(self):
      optimizer, criteria = self.create_model(self.device)
      if(self.args.resume):
        self.model = self.resume(device = self.device , optimizer = optimizer, model = self.model)
      self.model = self.train(model = self.model , optimizer = optimizer, criteria = criteria , train_loader = self.train_loader , test_loader = self.test_loader)
      return self.model

class PruneIt:

    def __init__(self,params):
                self.args = params
                if (self.args.arch == 'vgg'):
                    model = models.vgg()
                elif (self.args.arch == 'resnet'):
                    #model = models.__dict__[self.args.arch](dataset=self.args.data, depth=self.args.depth)
                     model = models.make_ns_resnet(num_classes = self.args.num_classes)
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model=model.to(self.device)

                ############################################################################
                """loading model from path"""

                if os.path.isfile(self.args.model):
                    print("=> loading checkpoint '{}'".format(self.args.model))
                    checkpoint = torch.load(self.args.model)
                    self.args.start_epoch = checkpoint['epoch']
                    best_prec1 = checkpoint['best_prec1']
                    model.load_state_dict(checkpoint['state_dict'])
                    print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(self.args.model, checkpoint['epoch'], best_prec1))
                self.model = model

    def threshold(self,model):
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
                thre_index = int(total * self.args.percent)
                thre = y[thre_index]   # the scaling threshold value, if a channel has scaling value lesser than threshold, it will be pruned
                return thre,total

    def config_create(self,model,thre,total):
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
                print(f"The pruned_ratio is {pruned_ratio}")
                return cfg,cfg_mask

    def model_to_be_pruned(self,cfg , device):
                device = self.device
                print(cfg)
                if (self.args.arch == 'vgg'):
                    newmodel = models.vgg()
                elif (self.args.arch == 'resnet'):
                    #newmodel = models.__dict__[self.args.arch](dataset=self.args.data, depth=self.args.depth)
                     newmodel = make_resnet_spl(self.args.num_classes , 32)
                newmodel=newmodel.to(device)
                print(len(cfg))
                return newmodel

    def prune(self,cfg,cfg_mask ,model , newmodel):

                ############################################################################
                """copying the weights corresponding to channels left after pruning in the new model"""

                if(self.args.arch=="vgg"):
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

                elif(self.args.arch=='resnet'):
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

                

                torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, self.args.save)
                print("The new model" , newmodel)
                
                self.model = newmodel
                return self.model

    def prune_model(self):
                thre,total = self.threshold(self.model)
                cfg,cfg_mask = self.config_create(self.model , thre ,total)
                newmodel = self.model_to_be_pruned(cfg,self.device)
                return self.prune(cfg,cfg_mask,self.model,newmodel)
                
            ##################################################################################
class OfNoUse:
  pass
class NetworkSlimming:
  def __init__(self ,data_loaders, path , save , model,   **kwargs):
    #super(NetworkSlimming,self).__init__(**kwargs)
    self.args = OfNoUse()
    self.kwargs = kwargs
    self.args.data = self.kwargs['NETWORK_SLIMMING']['GENERAL'].get('DATA' ,None)
    self.args.num_classes = self.kwargs['NETWORK_SLIMMING']['GENERAL'].get('NUM_CLASSES' , 10)      # dataset on which model is trained
    self.args.sparsity_reg = self.kwargs['NETWORK_SLIMMING']['PRETRAIN'].get('SPARSITY_REG' , True)    # true if training is done with sparsity regularization
    self.args.thr = self.kwargs['NETWORK_SLIMMING']['PRETRAIN'].get('THR' ,1e-5)      # the sparsity regularization hyperparameter value
    self.args.train_loader = data_loaders['train']
    self.args.test_loader = data_loaders['test']
    self.args.fine_tune = self.kwargs['NETWORK_SLIMMING']['PRETRAIN'].get('FINE_TUNE' ,False)        # true if pruned model is being fine-tuned
    self.args.path = self.kwargs['NETWORK_SLIMMING'].get('PATH' ,None)      # path from where the pruned model is loaded
    self.args.resume = self.kwargs['NETWORK_SLIMMING']['PRETRAIN'].get('RESUME' ,False)      # true of we have to resume training of some model whose checkpoint is saved
    self.args.train_bs = self.kwargs['NETWORK_SLIMMING']['PRETRAIN']['BATCH_SIZE'].get('TRAIN_BS' ,64)      # training batch size
    self.args.test_bs = self.kwargs['NETWORK_SLIMMING']['PRETRAIN']['BATCH_SIZE'].get('TEST_BS' ,256)        # test batch size
    self.args.epochs = self.kwargs['NETWORK_SLIMMING']['PRETRAIN'].get('EPOCHS' ,100)
    self.args.optimizer_name = self.kwargs['NETWORK_SLIMMING']['GENERAL'].get('OPTIMIZER_NAME', None)
    self.args.lr = self.kwargs['NETWORK_SLIMMING']['GENERAL'].get('LR' ,1e-1)
    self.args.momentum = self.kwargs['NETWORK_SLIMMING']['GENERAL'].get('MOMENTUM' ,0.9)
    self.args.weight_decay = self.kwargs['NETWORK_SLIMMING']['GENERAL'].get('WEIGHT_DECAY' ,1e-4)
    self.args.log_interval = self.kwargs['NETWORK_SLIMMING'].get('LOG_INTERVAL' , 100)     # number of intervals after which accuracy and loss values are printed during training
    self.args.arch = self.kwargs['NETWORK_SLIMMING']['GENERAL'].get('ARCH' ,'vgg')      # model architecture
    self.args.depth = self.kwargs['NETWORK_SLIMMING']['GENERAL'].get('DEPTH' , 164) 
    self.args.percent = self.kwargs['NETWORK_SLIMMING']['GENERAL'].get('PERCENT' , 0.6)
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
    model = x1.train_model()
    return model

  def prune(self):
    assert(self.args.save is not None)
    assert(self.args.model is not None)
    x1 = PruneIt(params=self.args)
    model = x1.prune_model()
    return model

  def compress(self):
    assert(self.args.fine_tune == True and self.args.sparsity_reg == False)
    x1 = Process(params=self.args)
    model = x1.train_model()
    return model

  def train_compress(self):
    assert(self.args.fine_tune == False)
    m1 = self.base_line()
    m1 = self.prune()
    self.args.sparsity_reg = False
    self.args.fine_tune = True
    return self.compress()
