import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from trailmet.algorithms.prune.prune import BasePruning
from trailmet.algorithms.prune.pns import SlimPruner
from trailmet.algorithms.prune.functional import update_bn_grad, summary_model
from tqdm import tqdm as tqdm_notebook

def build_model(net, num_classes=10):
    if net in ["resnet18", "resnet34", "resnet50"]:
        model = getattr(models, net)(num_classes=num_classes)
        # to get better result on cifar10
        model.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = torch.nn.Identity()
    elif net in ["mobilenet_v2"]:
        # to get better result on cifar10
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 1],
            [6, 64, 4, 1],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        model = getattr(models, net)(
            num_classes=num_classes, inverted_residual_setting=inverted_residual_setting
        )
        model.features[0][0].stride = (1, 1)
    else:
        raise NotImplementedError(f"{net}")

    return model

def adjust_learning_rate(optimizer, epoch, num_epochs, scheduler_type, lr):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    if scheduler_type==1:
        new_lr = lr * (0.3 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        if epoch in [num_epochs*0.5, num_epochs*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

class Network_Slimming(BasePruning):
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
        self.num_classes = kwargs.get('num_classes',100)
        self.pr = kwargs.get('pr',None)
        self.ft_only = kwargs.get('ft_only',False)
        self.scheduler_type = kwargs.get('scheduler_type', 1)
        self.weight_decay = kwargs.get('weight_decay' , 5e-4)
        self.net = kwargs.get('net')
        self.dataset = kwargs.get('dataset')
        self.epochs = kwargs.get('epochs',200)
        self.s = kwargs.get('s' , 1e-3)
        self.lr = kwargs.get('learning_rate',2e-3)
        self.prune_schema = os.path.join(kwargs.get('schema_root') , f"schema/{self.net}.json")
        self.sparsity_train = kwargs.get('sparsity_train',True)
        self.fine_tune_epochs = kwargs.get('fine_tune_epochs',165)
        self.fine_tune_lr = kwargs.get('fine_tune_learning_rate',1e-4)
        self.prune_ratio = kwargs.get('prune_ratio',0.5)
        self.log_base = f"{self.net}_{self.dataset}.pth"
        
    def compress_model(self,dataloaders) -> None:
        """Template function to be overwritten for each model compression method"""
        if(self.ft_only):
            print("Error")
            return 0
        model = Model(self.net,self.num_classes)
        self.log_name = self.log_base
        model = self.base_train(model,dataloaders,fine_tune = False)
        self.log_name = f"{self.net}_{self.dataset}_{self.prune_ratio}_.pth"
        self.lr = self.fine_tune_lr
        pruner = SlimPruner(model,self.prune_schema)
        pruning_result = pruner.run(self.prune_ratio)
        summary_model(pruner.pruned_model)
        pruned_model = pruner.pruned_model
        self.pr = pruning_result
        pruned_model.is_pruned = True
        del model
        pruned_model = self.base_train(pruned_model, dataloaders, fine_tune = True)        

    def update_bn_grad(self,model, s=0.0001):

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(s * torch.sign(m.weight.data))
    def base_train(self, model, dataloaders,fine_tune = False):
        """
        This function is used to perform standard model training and can be used for various purposes, such as model
        pretraining, fine-tuning of compressed models, among others. For cases, where base_train is not directly
        applicable, feel free to overwrite wherever this parent class is inherited.
        """
        num_epochs = self.fine_tune_epochs if fine_tune else self.epochs
        best_acc = 0    # setting to lowest possible value
        lr = self.lr if fine_tune else self.fine_tune_lr
        scheduler_type = self.scheduler_type
        weight_decay = self.weight_decay

        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        ###########################
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        train_losses = []
        valid_losses = []
        valid_accuracy = []
        
 
        for epoch in range(num_epochs):
            print("EPOCH NUMBER " , epoch)
            adjust_learning_rate(optimizer, epoch, num_epochs, scheduler_type, lr)
            t_loss = self.train_one_epoch(model, dataloaders['train'], criterion, optimizer)
            if self.sparsity_train:
                update_bn_grad(model)
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
                    }, self.log_name)
                else:
                    print("**Saving model**")
                    best_acc=acc
                    torch.save({
                        "epoch": epoch + 1,
                        "state_dict" : model.state_dict(),
                        "acc" : best_acc,
                    }, self.log_name)                       

            train_losses.append(t_loss)
            valid_losses.append(v_loss)
            valid_accuracy.append(acc)
            df_data=np.array([train_losses, valid_losses, valid_accuracy]).T
            df = pd.DataFrame(df_data, columns = ['train_losses','valid_losses','valid_accuracy'])
            df.to_csv(f'logs/{self.log_name}.csv')

        state = torch.load(self.log_name)
        model.load_state_dict(state['state_dict'],strict=True)
        return model 

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

class Model(nn.Module):
    def __init__(self,net, num_classes):
        super(Model,self).__init__()
        self.model = build_model(net , num_classes=num_classes)
    def forward(self,x):
        return self.model(x)

