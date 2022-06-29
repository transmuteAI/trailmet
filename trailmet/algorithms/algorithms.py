from .utils import adjust_learning_rate
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm as tqdm_notebook
class BaseAlgorithm(object):

    def __init__(self, **kwargs):

        self.pretraining_epochs = 200
        self.cuda_id = kwargs.get('cuda_id', 0) 
        self.device = torch.device(f"cuda:{str(self.cuda_id)}")
        self.log_name = kwargs.get('log_dir', 'abc')
        if os.path.exists('logs') == False:
            os.mkdir('logs')

        if os.path.exists('checkpoints') == False:
            os.mkdir('checkpoints')

    def compress_model(self):
        pass

    def pretrain(self, model, dataloaders, **kwargs):
        best_acc = 0
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
        
        if test_only == False:
            for epoch in range(num_epochs):
                adjust_learning_rate(optimizer, epoch, num_epochs, scheduler_type, lr)
                t_loss = self.train_one_epoch(model, dataloaders['train'], criterion, optimizer)
                acc, v_loss = self.test(model, dataloaders['val'], criterion)
                if acc>best_acc:
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

        state = torch.load(f"checkpoints/{self.log_name}_pretrained.pth")
        model.load_state_dict(state['state_dict'],strict=True)
        acc, v_loss = self.test(model, dataloaders['test'], criterion)
        print(f"Test Accuracy: {acc} | Valid Accuracy: {state['acc']}")

    

    def get_optimizer(self, optimizer_name, model, lr, weight_decay):
        if optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Unknown optimizer: %s' % optimizer_name)
        return optimizer

    def train_one_epoch(self, model, dataloader, loss_fn, optimizer, extra_functionality = None):
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

