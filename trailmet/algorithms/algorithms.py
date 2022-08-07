from .utils import adjust_learning_rate
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm as tqdm_notebook
class BaseAlgorithm(object):
    """
    Base Algorithm class that defines the structure of each model compression algorithm implemented in this library.
    Every new algorithm is expected to directly use or overwrite the template functions defined below.

    The root command to invoke the compression of any model is .compress_model(). Thus, it is required that all
    algorithms complete this template function and use it as the first point of invoking the model compression process.

    For methods that require to perform pretraining and fine-tuning, the implementation of base_train() method can
    directly be used for both the tasks. In case of modifications, overwrite this function based on the needs.
    """
    def __init__(self, **kwargs):

        self.pretraining_epochs = 200
        self.cuda_id = kwargs.get('cuda_id', 0) 
        self.device = torch.device(f"cuda:{str(self.cuda_id)}")
        self.log_name = kwargs.get('log_dir', 'abc')
        if os.path.exists('logs') is False:
            os.mkdir('logs')

        if os.path.exists('checkpoints') is False:
            os.mkdir('checkpoints')

    def compress_model(self) -> None:
        """Template function to be overwritten for each model compression method"""
        pass

    def base_train(self, model, dataloaders, **kwargs) -> None:
        """
        This function is used to perform standard model training and can be used for various purposes, such as model
        pretraining, fine-tuning of compressed models, among others. For cases, where base_train is not directly
        applicable, feel free to overwrite wherever this parent class is inherited.
        """
        best_acc = 0    # setting to lowest possible value
        num_epochs = kwargs.get('EPOCHS', self.pretraining_epochs)
        test_only = kwargs.get('TEST_ONLY', False)

        ### preparing optimizer ###
        optimizer_name = kwargs.get('OPTIMIZER', 'SGD')
        lr = kwargs.get('LR', 0.05)
        scheduler_type = kwargs.get('SCHEDULER_TYPE', 1)
        weight_decay = kwargs.get('WEIGHT_DECAY', 0.001)

        optimizer = self.get_optimizer(optimizer_name, model, lr, weight_decay)
        ###########################

        criterion = nn.CrossEntropyLoss()
        train_losses = []
        valid_losses = []
        valid_accuracy = []
        
        if test_only is False:
            for epoch in range(num_epochs):
                adjust_learning_rate(optimizer, epoch, num_epochs, scheduler_type, lr)
                t_loss = self.train_one_epoch(model, dataloaders['train'], criterion, optimizer)
                acc, v_loss = self.test(model, dataloaders['val'], criterion)
                if acc > best_acc:
                    print("**Saving model**")
                    best_acc=acc
                    torch.save({
                        "epoch": epoch + 1,
                        "state_dict" : model.state_dict(),
                        "acc" : best_acc,
                    }, f"checkpoints/{self.log_name}.pth")
                train_losses.append(t_loss)
                valid_losses.append(v_loss)
                valid_accuracy.append(acc)
                df_data=np.array([train_losses, valid_losses, valid_accuracy]).T
                df = pd.DataFrame(df_data, columns = ['train_losses','valid_losses','valid_accuracy'])
                df.to_csv(f'logs/{self.log_name}.csv')

        state = torch.load(f"checkpoints/{self.log_name}.pth")
        model.load_state_dict(state['state_dict'],strict=True)
        acc, v_loss = self.test(model, dataloaders['test'], criterion)
        print(f"Test Accuracy: {acc} | Valid Accuracy: {state['acc']}")

    

    def get_optimizer(self, optimizer_name: str, model, lr, weight_decay):
        """returns the optimizer with the given name"""
        if optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
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
                if len(images)!=64:      # To do : fix this
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
        if loss_fn is not None:
            return running_acc1/counter, running_acc5/counter, running_loss/counter
        return running_acc1/counter, running_acc5/counter
