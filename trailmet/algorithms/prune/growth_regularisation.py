#!/usr/bin/env python
# coding: utf-8
# %%


import argparse
import json
import os
pjoin = os.path.join
import numpy as np
import pandas as pd
import sys
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from trailmet.algorithms.utils import strlist_to_list, strdict_to_dict, check_path,merge_args,PresetLRScheduler
from trailmet.algorithms.prune.prune import BasePruning
from trailmet.algorithms.prune.pns import SlimPruner
from trailmet.algorithms.prune.functional import update_bn_grad, summary_model
from trailmet.algorithms.utils import Logger
from trailmet.algorithms.prune.pruner import pruner_dict


from copy import deepcopy
from tqdm import tqdm as tqdm_notebook
import numpy as np
from importlib import import_module



# %%


def adjust_learning_rate(optimizer, epoch, num_epochs, scheduler_type, lr):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    if scheduler_type==1:
        new_lr = lr * (0.3 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        if epoch in [num_epochs*0.5, num_epochs*0.75]:
            for param_group in optimizer.param_groups:
                pa
                ram_group['lr'] *= 0.1


# %%
def is_single_branch(name):
    return False

# %%


class Growth_Regularisation(BasePruning):
    """
    Base Algorithm class that defines the structure of each model compression algorithm implemented in this library.
    Every new algorithm is expected to directly use or overwrite the template functions defined below.
    The root command to invoke the compression of any model is .compress_model(). Thus, it is required that all
    algorithms complete this template function and use it as the first point of invoking the model compression process.
    For methods that require to perform pretraining and fine-tuning, the implementation of base_train() method can
    directly be used for both the tasks. In case of modifications, overwrite this function based on the needs.
    """
    def __init__(self,**kwargs):
        self.device = 'cuda'
        if os.path.exists('logs') is False:
            os.mkdir('logs')
#         if os.path.exists('checkpoints') is False:
#             os.mkdir('checkpoints_1')
        class cfg:
            data = ''
            dataset = 'cifar100'
            arch = 'resnet50'
            workers = 2
            epochs = 125
            start_epoch = 0
            batch_size = 256
            lr = 1e-1
            momentum = 0.9
            weight_decay = 1e-4
            wd = 1e-4
            print_freq = 100
            prune_ratio = 0.5
            log_name = 'r50_50_c100'
            resume = ''
            evaluate = True
            pretrained = False
            world_size = -1
            rank = -1
            dist_url = None
            dist_backend = None
            seed = 42
            gpu = 0

            project_name = "TrAIL"
            debug = False
            screen_print = False
            note = " "
            print_interval = 100
            test_interval = 2000
            plot_interval = 1e+8
            save_interval = 2000
            params_json = ''

            resume_path = None
            directly_ft_weights = None
            base_model_path = 'r50_50_c100.pth'
            start_epoch = 0

            method = 'GReg-1'
            stage_pr = "[0.5,0.5,0.5,0.5,0.5,0.5]"
            skip_layers = ""
            lr_ft = {"0":0.01,"60":0.001,"90":0.0001}
            data_path = "./data"
            wg = "filter"
            pick_pruned = "min"
            reinit = ""
            use_bn = True
            block_loss_grad = False
            save_mag_reg_log = False
            save_order_log = False
            mag_ratio_limit = 1000
            base_pr_model = None
            inherit_pruned = 'index'
            model_noise_std =  0
            model_noise_sum = 10
            orcal_pruning = False
            ft_in_oracle_pruning = False
        #     check_jsv_loop = 0

            batch_size_prune = 256
            lr_prune = 1e-3
            update_reg_interval = 1
            stabilize_reg_interval = 1
            reg_upper_limit = 1e-4
            reg_upper_limit_pick = 1e-2
            reg_granularity_pick = 1e-5
            reg_granularity_prune = 2e-4
            reg_granularity_recover = 1e-4

            copy_bn_w = True
            copy_bn_b = True
            reg_multiplier = 1
        self.args = cfg()
        self.args.arch = kwargs.get('arch','resnet50')
        self.args.num_classes = kwargs.get('num_classes',100)
        self.args.dataset = kwargs.get('dataset','cifar100')
        self.args.epochs = kwargs.get('epochs',125)
        self.args.lr = kwargs.get('lr',1e-1)
        self.args.momentum = kwargs.get('momentum',0.9)
        self.args.weight_decay = kwargs.get('weight_decay',1e-4)
        self.args.wd = kwargs.get('wd',1e-4)
        self.args.prune_ratio = kwargs.get('prune_ratio',0.5)
        self.args.method = 'GReg-1'
        self.args.stage_pr = kwargs.get('stage_pr',"[0.5,0.5,0.5,0.5,0.5,0.5]")
        self.args.reg_upper_limit = kwargs.get('reg_upper_limit', 1e-4)
        self.args.reg_upper_pick = kwargs.get('reg_upper_pick', 1e-2)
        self.args.reg_granularity_pick  = kwargs.get('reg_granularity_pick' , 1e-5)
        self.args.reg_granularity_prune = kwargs.get('reg_granularity_prune',2e-4)
        self.args.reg_granularity_recover = kwargs.get('reg_granularity_recover',1e-4)
        global logger  
        logger = Logger(self.args)
        global logprint
        logprint = logger.log_printer.logprint
        global accprint
        accprint = logger.log_printer.accprint
        global netprint
        netprint = logger.netprint
        if self.args.stage_pr:
#             if is_single_branch(self.args.arch): # e.g., alexnet, vgg
#                 self.args.stage_pr = parse_prune_ratio_vgg(self.args.stage_pr, num_layers=num_layers[self.args.arch]) # example: [0-4:0.5, 5:0.6, 8-10:0.2]
#                 self.args.skip_layers = strlist_to_list(self.args.skip_layers, str) # example: [0, 2, 6]
#             else: # e.g., resnet
            self.args.stage_pr = strlist_to_list(self.args.stage_pr, float) # example: [0, 0.4, 0.5, 0]
            self.args.skip_layers = strlist_to_list(self.args.skip_layers, str) # example: [2.3.1, 3.1]
        else:
            assert self.args.base_pr_model, 'If stage_pr is not provided, base_pr_model must be provided'
            

        
        
        
    def compress_model(self,dataloaders) -> None:
        """Template function to be overwritten for each model compression method"""

        self.args.epochs = 1
        model = getattr(models, self.args.arch)(num_classes=self.args.num_classes)
        model.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = torch.nn.Identity()
        model = self.base_train(self.args,model,dataloaders,fine_tune = False)
        self.args.log_name = f"{self.args.arch}_{self.args.dataset}_{self.args.prune_ratio}_pruned"
        self.prune_and_finetune(self.args,dataloaders)

    def prune_and_finetune(self,args,dataloader):
        model = getattr(models, args.arch)(num_classes=args.num_classes)
        # to get better result on cifar10
        model.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = torch.nn.Identity()
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        
        if args.base_model_path:
#         ckpt = torch.load(args.base_model_path)
#         if 'model' in ckpt:
#             model = ckpt['model']
#         model.load_state_dict(ckpt['state_dict'])
            X = torch.load(args.base_model_path)
            X1 = X['state_dict']
            L = list(X1.keys())
            for key in L:
                new_key = key.replace('model.','')
                X1[new_key] = X1.pop(key)
            model.load_state_dict(X1)
            #print(validate(val_loader,  model, nn.CrossEntropyLoss().cuda(args.gpu),args))
            logprint("==> Load pretrained model successfully: '%s'" % args.base_model_path)
        
        
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay) 
        prune_state, pruner = '', None
        if prune_state != 'finetune':
            class passer: pass # to pass arguments
#             passer.test = validate
#             passer.finetune = finetune
            passer.train_loader = dataloader['train']
            passer.test_loader = dataloader['val']
            passer.save = self.save_model
            passer.criterion = criterion
            passer.train_sampler = None
            passer.pruner = pruner
            passer.args = args
            passer.is_single_branch = is_single_branch
            pruner = pruner_dict[args.method].Pruner(model, args, logger, passer)
            if(args.method == 'L1'):
                model  = pruner.prune()
                print("**Saving model without key**")
                print(model)
                torch.save({
                            "arch" : args.arch,
                            "model" : model,
                            "state_dict" : model.state_dict(),
                            }, f"{args.log_name}.pth")  
            else:
                pruning_key , model = pruner.prune() # get the pruned model
                #print(model)
                print("**Saving model with key**")
                print(model)
                torch.save({
                            "arch" : args.arch,
                            "model" : model,
                            "state_dict" : model.state_dict(),
                            "pruning_key" : pruning_key,
                            }, f"{args.log_name}.pth") 
        self.base_train(args,model,dataloader,pruning_key)
        

    def base_train(self, args, model, dataloaders,fine_tune = False):
        """
        This function is used to perform standard model training and can be used for various purposes, such as model
        pretraining, fine-tuning of compressed models, among others. For cases, where base_train is not directly
        applicable, feel free to overwrite wherever this parent class is inherited.
        """
        num_epochs = args.epochs
        best_acc = 0    # setting to lowest possible value
        lr = args.lr 
        scheduler_type = 1
        weight_decay = 1e-4
        self.pr = None
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        ###########################
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        train_losses = []
        valid_losses = []
        valid_accuracy = []
        
 
        for epoch in range(args.epochs):
            print("EPOCH NUMBER " , epoch)
            adjust_learning_rate(optimizer, epoch, num_epochs, scheduler_type, lr)
            t_loss = self.train_one_epoch(model, dataloaders['train'], criterion, optimizer)
      
            acc, v_loss = self.test(model, dataloaders['val'], criterion)
            if acc > best_acc:
                if self.pr is not None:  
                    print("**Saving model with key**")
                    best_acc=acc
                    torch.save({
                        "epoch": epoch + 1,
                        "state_dict" : model.state_dict(),
                        "pruning_key" : self.pr,
                        "acc" : best_acc,
                    }, f"{args.log_name}.pth")
                else:
                    print("**Saving model**")
                    best_acc=acc
                    torch.save({
                        "epoch": epoch + 1,
                        "state_dict" : model.state_dict(),
                        "acc" : best_acc,
                    }, f"{args.log_name}.pth")                       

            train_losses.append(t_loss)
            valid_losses.append(v_loss)
            valid_accuracy.append(acc)
            df_data=np.array([train_losses, valid_losses, valid_accuracy]).T
            df = pd.DataFrame(df_data, columns = ['train_losses','valid_losses','valid_accuracy'])
            df.to_csv(f'logs/{args.log_name}.csv')

    

    def get_optimizer(self, optimizer_name: str, model, lr, weight_decay):
        """returns the optimizer with the given name"""
        if optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Unknown optimizer: %s' % optimizer_name)
        return optimizer

    def train_one_epoch(self, model, dataloader, loss_fn, optimizer, extra_functionality = None):
        """standard training loop which can be used for various purposes with an extra functionality function to add to its working at the end of the loop."""
        model.train()
        counter = 0
        tk1 = tqdm_notebook(dataloader, total=len(dataloader))
        running_loss = 0
        for x_var, y_var in tk1:
            counter +=1
            x_var = x_var.to(device=self.device)
            y_var = y_var.to(device=self.device)
            scores = model(x_var)

            loss = loss_fn(scores, y_var)
            running_loss+=loss.item()
            tk1.set_postfix(loss=running_loss/counter)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if extra_functionality is not None:
                extra_functionality()
        return running_loss/counter

    def test(self, model, dataloader, loss_fn):
        """This method is used to test the performance of the trained model."""
        model.eval()
        counter = 0
        tk1 = tqdm_notebook(dataloader, total=len(dataloader))
        running_loss = 0
        running_acc = 0
        total = 0
        with torch.no_grad():
            for x_var, y_var in tk1:
                counter +=1
                x_var = x_var.to(device=self.device)
                y_var = y_var.to(device=self.device)
                scores = model(x_var)
                loss = loss_fn(scores, y_var)
                _, scores = torch.max(scores.data, 1)
                y_var = y_var.cpu().detach().numpy()
                scores = scores.cpu().detach().numpy()

                correct = (scores == y_var).sum().item()
                running_loss+=loss.item()
                running_acc+=correct
                total+=scores.shape[0]
                tk1.set_postfix(loss=running_loss/counter, acc=running_acc/total)
        return running_acc/total, running_loss/counter
    def save_model(self,state, is_best=False, mark=''):
        print(logger.weights_path , state['acc1'] , state['acc5'] )
        out = pjoin(logger.weights_path, "checkpoint.pth")
        torch.save(state, out)
        if is_best:
            out_best = pjoin(logger.weights_path, "checkpoint_best.pth")
            torch.save(state, 'ft_r50_9375_c100.pth')
        if mark:
            out_mark = pjoin(logger.weights_path, "checkpoint_{}.pth".format(mark))
            torch.save(state, out_mark)

