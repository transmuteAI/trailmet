import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm as tqdm_notebook

from trailmet.algorithms.distill.distill import Distillation, ForwardHookManager

# Paraphrasing Complex Network: Network Compression via Factor Transfer

class FactorTransferLoss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.criterion = nn.L1Loss()
        self.criterion_ce = nn.CrossEntropyLoss()
        
    @staticmethod
    def FT(x):
        return F.normalize(x.view(x.size(0), -1))
    
    def forward(self, factor_teacher, factor_student, logits, labels):
        loss = self.criterion_ce(logits, labels)
        loss += self.beta * self.criterion(self.FT(factor_student), self.FT(factor_teacher.detach()))
        return loss

class Paraphraser(nn.Module):
    def __init__(self,in_planes, planes, stride=1):
        super(Paraphraser, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.1)
#       self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv0 = nn.Conv2d(in_planes,in_planes , kernel_size=3, stride=1, padding=1, bias=True)
#       self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
#       self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
#       self.bn0_de = nn.BatchNorm2d(planes)
        self.deconv0 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
#       self.bn1_de = nn.BatchNorm2d(in_planes)
        self.deconv1 = nn.ConvTranspose2d(planes,in_planes, kernel_size=3, stride=1, padding=1, bias=True)
#       self.bn2_de = nn.BatchNorm2d(in_planes)
        self.deconv2 = nn.ConvTranspose2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)

#### Mode 0 - throw encoder and decoder (reconstruction)
#### Mode 1 - extracting teacher factors
    def forward(self, x,mode):

        if mode == 0:
            ## encoder
            out = self.leakyrelu((self.conv0(x)))
            out = self.leakyrelu((self.conv1(out)))
            out = self.leakyrelu((self.conv2(out)))
            ## decoder
            out = self.leakyrelu((self.deconv0(out)))
            out = self.leakyrelu((self.deconv1(out)))
            out = self.leakyrelu((self.deconv2(out)))


        if mode == 1:
            out = self.leakyrelu((self.conv0(x)))
            out = self.leakyrelu((self.conv1(out)))
            out = self.leakyrelu((self.conv2(out)))

        ## only throw decoder
        if mode == 2:
            out = self.leakyrelu((self.deconv0(x)))
            out = self.leakyrelu((self.deconv1(out)))
            out = self.leakyrelu((self.deconv2(out)))
        return out


class Translator(nn.Module):
    def __init__(self,in_planes, planes, stride=1):
        super(Translator, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.1)
#       self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv0 = nn.Conv2d(in_planes,in_planes , kernel_size=3, stride=1, padding=1, bias=True)
#       self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
#       self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        out = self.leakyrelu((self.conv0(x)))
        out = self.leakyrelu((self.conv1(out)))
        out = self.leakyrelu((self.conv2(out)))
        return out


class FactorTransfer(Distillation):
    """class to compress model using distillation via attention transfer"""
    def __init__(self, teacher_model, student_model, dataloaders, paraphraser=None, **kwargs):
        super(FactorTransfer, self).__init__(**kwargs)
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.paraphraser = paraphraser
        self.dataloaders = dataloaders
        self.kwargs = kwargs
        self.device = kwargs['DEVICE']
        self.beta = self.kwargs['DISTILL_ARGS'].get('BETA', 500);
        self.verbose = self.kwargs['VERBOSE']

        #self.student_io_dict, self.teacher_io_dict = dict(), dict()
        self.teacher_layer_name = kwargs['DISTILL_ARGS'].get('TEACHER_LAYER_NAME')
        self.student_layer_name = kwargs['DISTILL_ARGS'].get('STUDENT_LAYER_NAME')
        self.forward_hook_manager_teacher = ForwardHookManager(self.device)
        self.forward_hook_manager_student = ForwardHookManager(self.device)

        self.ft_loss = FactorTransferLoss(self.beta)
        self.l1_loss = nn.L1Loss()

    def compress_model(self):
        """function to transfer knowledge from teacher to student"""

        # include teacher training options

        self.register_hooks()

        if self.paraphraser == None:
            if 'paraphraser' in self.dataloaders:
                self.train_paraphraser(self.teacher_model, self.dataloaders['paraphraser'], **self.kwargs['PARAPHRASER'])
            else:
                self.train_paraphraser(self.teacher_model, self.dataloaders, **self.kwargs['PARAPHRASER'])

        self.distill(self.teacher_model, self.student_model, self.paraphraser, self.dataloaders, **self.kwargs['DISTILL_ARGS'])

    def distill(self, teacher_model, student_model, paraphraser, dataloaders, **kwargs):
        if self.verbose:
            print("=====TRAINING STUDENT NETWORK=====")

        self.register_hooks()
        num_epochs = kwargs.get('EPOCHS', 163)
        test_only = kwargs.get('TEST_ONLY', False)
        lr = kwargs.get('LR', 0.1)
        weight_decay = kwargs.get('WEIGHT_DECAY', 0.0005)
        
        in_planes = kwargs.get('IN_PLANES', 64)
        rate = kwargs.get('RATE', 0.5)
        planes = kwargs.get('planes', int(in_planes*rate))

        translator = Translator(in_planes, planes)
        translator.to(self.device)
        
        # dont hard code this
        optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[82, 123], gamma=0.1, verbose=False)
        optimizer_translator = torch.optim.SGD(translator.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        scheduler_translator = torch.optim.lr_scheduler.MultiStepLR(optimizer_translator, milestones=[82, 123], gamma=0.1, verbose=False)
        
        criterion = self.criterion
        
        best_acc = 0
        train_losses = []
        valid_losses = []
        valid_accuracy = []

        if test_only == False:
            for epoch in range(num_epochs):
                if self.verbose:
                    print(f"Epoch: {epoch+1}")
                t_loss = self.train_one_epoch(teacher_model, student_model, paraphraser, translator,
                                              dataloaders['train'], criterion, optimizer, optimizer_translator)
                acc, v_loss = self.test(teacher_model, student_model, paraphraser, translator,
                                        dataloaders['val'], criterion)
                
                # use conditions for different schedulers e.g. ReduceLROnPlateau needs scheduler.step(v_loss)
                scheduler.step()
                scheduler_translator.step()
                
                if acc > best_acc:
                    if self.verbose:
                        print("**Saving checkpoint**")
                    best_acc = acc
                    torch.save({
                        "epoch": epoch+1,
                        "state_dict": student_model.state_dict(),
                        "accuracy": acc,
                    }, f"checkpoints/{self.log_name}.pth")
                train_losses.append(t_loss)
                valid_losses.append(v_loss)
                valid_accuracy.append(acc)
                df_data=np.array([train_losses, valid_losses, valid_accuracy]).T
                df = pd.DataFrame(df_data, columns = ['train_losses','valid_losses','valid_accuracy'])
                df.to_csv(f'logs/{self.log_name}.csv')

    
    def train_one_epoch(self, teacher_model, student_model, paraphraser, translator, dataloader, loss_fn, optimizer, optimizer_translator):
        teacher_model.eval()
        paraphraser.eval()
        student_model.train()
        translator.train()

        counter = 0
        running_loss = 0
        tk1 = tqdm_notebook(dataloader, total=len(dataloader))

        for (images, labels) in tk1:
            counter += 1
            images = images.to(self.device, dtype=torch.float)
            labels = labels.to(self.device)
            
            teacher_preds = teacher_model(images)
            student_preds = student_model(images)

            teacher_io_dict = self.forward_hook_manager_teacher.pop_io_dict()
            student_io_dict = self.forward_hook_manager_student.pop_io_dict()
            feature_map_pair = [teacher_io_dict[self.teacher_layer_name]['output'],
                                student_io_dict[self.student_layer_name]['output']]
            
            teacher_factor = paraphraser(feature_map_pair[0], mode=1)
            student_factor = translator(feature_map_pair[1])

            loss = loss_fn(teacher_factor, student_factor, teacher_preds, student_preds, labels)
            running_loss+=loss.item()
            tk1.set_postfix(loss=running_loss/counter)
            optimizer.zero_grad()
            optimizer_translator.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_translator.step()

        return (running_loss/counter)

    def test(self, teacher_model, student_model, paraphraser, translator, dataloader, loss_fn):
        teacher_model.eval()
        paraphraser.eval()
        student_model.eval()
        translator.eval()

        counter = 0
        total = 0
        running_loss = 0
        running_acc = 0
        tk1 = tqdm_notebook(dataloader, total=len(dataloader))

        for (images, labels) in tk1:
            counter += 1
            images = images.to(self.device, dtype=torch.float)
            labels = labels.to(self.device)
            
            with torch.no_grad():
                teacher_preds = teacher_model(images)
                student_preds = student_model(images)

            teacher_io_dict = self.forward_hook_manager_teacher.pop_io_dict()
            student_io_dict = self.forward_hook_manager_student.pop_io_dict()
            feature_map_pair = [teacher_io_dict[self.teacher_layer_name]['output'],
                                student_io_dict[self.student_layer_name]['output']]
            
            with torch.no_grad():
                teacher_factor = paraphraser(feature_map_pair[0], mode=1)
                student_factor = translator(feature_map_pair[1])

            loss = loss_fn(teacher_factor, student_factor, teacher_preds, student_preds, labels)
            running_loss+=loss.item()

            preds = student_preds.softmax(1).to('cpu').numpy()
            valid_labels = labels.to('cpu').numpy()
            running_acc += (valid_labels == preds.argmax(1)).sum().item()
            total+=preds.shape[0]
            tk1.set_postfix(loss=running_loss/counter, acc = running_acc/total)
        return (running_acc/total), (running_loss/counter)
    
    def criterion(self, teacher_factor, student_factor, teacher_preds, student_preds, labels):
        return self.ft_loss(teacher_factor, student_factor, student_preds, labels)

    def train_paraphraser(self, teacher_model, dataloaders, **kwargs):
        
        in_planes = kwargs.get('IN_PLANES', 64)
        rate = kwargs.get('RATE', 0.5)
        planes = kwargs.get('PLANES', int(in_planes*rate))
        paraphraser = Paraphraser(in_planes, planes)
        paraphraser.to(self.device)
        
        path = kwargs.get('PATH', '')
        if path != '':
            if self.verbose:
                print("=====LOADING PARAPHRASER=====")
            paraphraser.load_state_dict(torch.load(path))
            self.paraphraser = paraphraser
        else:
            if self.verbose:
                print("=====TRAINING PARAPHRASER=====")
            num_epochs = kwargs.get('EPOCHS', 5)
            lr = kwargs.get('LR', 0.1)
            weight_decay = kwargs.get('WEIGHT_DECAY', 0.0005)

            optimizer = torch.optim.SGD(paraphraser.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
            criterion = self.l1_loss

            paraphraser.train()
            for epoch in range(num_epochs):
                t_loss = self.train_one_epoch_paraphraser(teacher_model, paraphraser, dataloaders['train'], criterion, optimizer)
                if self.verbose:
                    print(f"Epoch {epoch+1} | Train loss: {t_loss:.4f}")

            torch.save(paraphraser.state_dict(), f'checkpoints/{self.log_name}_paraphraser.pth')
            self.paraphraser = paraphraser
            self.paraphraser.load_state_dict(torch.load(f'checkpoints/{self.log_name}_paraphraser.pth'))

    def train_one_epoch_paraphraser(self, teacher_model, paraphraser, dataloader, criterion, optimizer):
        teacher_model.eval()
        paraphraser.train()
        
        counter = 0
        running_loss = 0
        tk1 = tqdm_notebook(dataloader, total=len(dataloader))

        for (images, labels) in tk1:
            counter += 1
            images = images.to(self.device, dtype=torch.float)
            labels = labels.to(self.device)
            
            teacher_preds = teacher_model(images)
            teacher_io_dict = self.forward_hook_manager_teacher.pop_io_dict()
            feature_map = teacher_io_dict[self.teacher_layer_name]['output']
            paraphraser_output = paraphraser(feature_map, mode=0)

            loss = criterion(paraphraser_output, feature_map.detach())
            running_loss+=loss.item()
            tk1.set_postfix(loss=running_loss/counter)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return (running_loss/counter)

    def register_hooks(self):
        self.forward_hook_manager_teacher.add_hook(self.teacher_model, self.teacher_layer_name, requires_input=False, requires_output=True)
        self.forward_hook_manager_student.add_hook(self.student_model, self.student_layer_name, requires_input=False, requires_output=True)
