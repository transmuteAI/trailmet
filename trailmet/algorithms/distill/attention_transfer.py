from tabnanny import verbose
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm as tqdm_notebook

from trailmet.algorithms.distill.distill import Distillation, ForwardHookManager

# Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer

class AttentionTransferLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def attention_map(feature_map):
        """Compute the attention map from a feature map"""
        return F.normalize(feature_map.pow(2).mean(1).flatten(1))

    def compute_loss(self, teacher_feature_map, student_feature_map):
        """Compute the loss between teacher and student feature maps"""
        teacher_attention_map = self.attention_map(teacher_feature_map)
        student_attention_map = self.attention_map(student_feature_map)
        loss = (teacher_attention_map - student_attention_map).pow(2).mean()
        return loss
    
    def forward(self, feature_map_pairs):
        """feature_map_pairs: list of (teacher_feature_map, student_feature_map)"""
        loss = 0
        for (teacher_feature_map, student_feature_map) in feature_map_pairs:
            loss += self.compute_loss(teacher_feature_map, student_feature_map)
        return loss

class AttentionTransfer(Distillation):
    """class to compress model using distillation via attention transfer"""
    def __init__(self, teacher_model, student_model, dataloaders, **kwargs):
        super(AttentionTransfer, self).__init__(**kwargs)
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.dataloaders = dataloaders
        self.kwargs = kwargs
        self.beta = self.kwargs['DISTILL_ARGS'].get('BETA', 1000);
        
        #self.student_io_dict, self.teacher_io_dict = dict(), dict()
        self.teacher_layer_names = kwargs['DISTILL_ARGS'].get('TEACHER_LAYER_NAMES')
        self.student_layer_names = kwargs['DISTILL_ARGS'].get('STUDENT_LAYER_NAMES')
        self.forward_hook_manager_teacher = ForwardHookManager(self.device)
        self.forward_hook_manager_student = ForwardHookManager(self.device)

        self.ce_loss = nn.CrossEntropyLoss()
        self.at_loss = AttentionTransferLoss()

    def compress_model(self):
        """function to transfer knowledge from teacher to student"""
        # include teacher training options
        self.distill(self.teacher_model, self.student_model, self.dataloaders, **self.kwargs['DISTILL_ARGS'])

    def distill(self, teacher_model, student_model, dataloaders, **kwargs):
        verbose = kwargs.get('VERBOSE', False)
        if verbose:
            print("=====TRAINING STUDENT NETWORK=====")

        self.register_hooks()
        num_epochs = kwargs.get('EPOCHS', 200)
        test_only = kwargs.get('TEST_ONLY', False)
        lr = kwargs.get('LR', 0.1)
        weight_decay = kwargs.get('WEIGHT_DECAY', 0.0005)
        milestones = kwargs.get('MILE_STONES', [82, 123])
        gamma = kwargs.get('GAMMA', 0.1)
        
        # dont hard code this
        optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, verbose=False)
        criterion = self.criterion
        
        best_acc = 0
        train_losses = []
        valid_losses = []
        valid_accuracy = []

        if test_only == False:
            for epoch in range(num_epochs):
                if verbose:
                    print(f"Epoch: {epoch+1}")
                t_loss = self.train_one_epoch(teacher_model, student_model, dataloaders['train'], criterion, optimizer)
                acc, v_loss = self.test(teacher_model, student_model, dataloaders['val'], criterion)
                
                # use conditions for different schedulers e.g. ReduceLROnPlateau needs scheduler.step(v_loss)
                scheduler.step()
                
                if acc > best_acc:
                    if verbose:
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

        
    def train_one_epoch(self, teacher_model, student_model, dataloader, loss_fn, optimizer):
        teacher_model.eval()
        student_model.train()

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
            feature_map_pairs = []
            for teacher_layer, student_layer in zip(self.teacher_layer_names, self.student_layer_names):
                feature_map_pairs.append(
                    (teacher_io_dict[teacher_layer]['output'], student_io_dict[student_layer]['output'])
                    )

            loss = loss_fn(teacher_preds, student_preds, feature_map_pairs, labels)
            running_loss+=loss.item()
            tk1.set_postfix(loss=running_loss/counter)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return (running_loss/counter)

    def test(self, teacher_model, student_model, dataloader, loss_fn):
        teacher_model.eval()
        student_model.eval()

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
            feature_map_pairs = []
            for teacher_layer, student_layer in zip(self.teacher_layer_names, self.student_layer_names):
                feature_map_pairs.append(
                    (teacher_io_dict[teacher_layer]['output'], student_io_dict[student_layer]['output'])
                    )

            loss = loss_fn(teacher_preds, student_preds, feature_map_pairs, labels)
            running_loss+=loss.item()

            preds = student_preds.softmax(1).to('cpu').numpy()
            valid_labels = labels.to('cpu').numpy()
            running_acc += (valid_labels == preds.argmax(1)).sum().item()
            total+=preds.shape[0]
            tk1.set_postfix(loss=running_loss/counter, acc = running_acc/total)
        return (running_acc/total), (running_loss/counter)
    
    def criterion(self, teacher_preds, student_preds, feature_map_pairs, labels):
        ce_loss = self.ce_loss(student_preds, labels)
        at_loss = self.at_loss(feature_map_pairs)
        return ce_loss + self.beta*at_loss
            
    def register_hooks(self):
        for layer in self.teacher_layer_names:
            self.forward_hook_manager_teacher.add_hook(self.teacher_model, layer, requires_input=False, requires_output=True)

        for layer in self.student_layer_names:
            self.forward_hook_manager_student.add_hook(self.student_model, layer, requires_input=False, requires_output=True)