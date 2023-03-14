
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm as tqdm_notebook

from trailmet.algorithms.distill.distill import Distillation
from trailmet.algorithms.distill.losses import KDTransferLoss
from trailmet.algorithms.distill.losses import RkdDistance
from trailmet.algorithms.distill.losses import RKdAngle



class RKDTransfer(Distillation):
    """class to compress model using distillation via RKD transfer"""
    def __init__(self, teacher_model, student_model, dataloaders, **kwargs):
        super(RKDTransfer, self).__init__(**kwargs)
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.dataloaders = dataloaders
        self.kwargs = kwargs
        self.device = kwargs['DEVICE']
        self.lambda_ = self.kwargs['DISTILL_ARGS'].get('BETA', {'lambda_d':25,'lambda_a':50})

        #self.student_io_dict, self.teacher_io_dict = dict(), dict()
        self.temperature=self.kwargs['DISTILL_ARGS'].get('TEMPERATURE',5)
        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = KDTransferLoss(self.temperature)
        self.distance_loss=RkdDistance()
        self.angle_loss=RKdAngle()

    def compress_model(self):
        """function to transfer knowledge from teacher model to student model"""
        # include teacher training options
        self.distill(self.teacher_model, self.student_model, self.dataloaders, **self.kwargs['DISTILL_ARGS'])

    def distill(self, teacher_model, student_model, dataloaders, **kwargs):
        print("=====TRAINING STUDENT NETWORK=====")
        #self.register_hooks()
        num_epochs = kwargs.get('EPOCHS', 200)
        test_only = kwargs.get('TEST_ONLY', False)
        lr = kwargs.get('LR', 0.1)
        weight_decay = kwargs.get('WEIGHT_DECAY', 0.0005)
        
        # dont hard code this
        optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2, verbose=False)
        criterion = self.criterion
        
        best_acc = 0
        train_losses = []
        valid_losses = []
        valid_accuracy = []

        if test_only == False:
            for epoch in range(num_epochs):
                print(f"Epoch: {epoch+1}")
                t_loss = self.train_one_epoch(teacher_model, student_model, dataloaders['train'], criterion, optimizer)
                acc, v_loss = self.test(teacher_model, student_model, dataloaders['val'], criterion)
                
                # use conditions for different schedulers e.g. ReduceLROnPlateau needs scheduler.step(v_loss)
                scheduler.step()
                
                if acc > best_acc:
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
            
            teacher_pool,teacher_preds = teacher_model.forward(images,is_feat=True)
            student_pool,student_preds = student_model.forward(images,is_feat=True)

            
            feature_map_pairs = []
            feature_map_pairs.append((teacher_pool,student_pool))

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
                teacher_pool,teacher_preds = teacher_model.forward(images,is_feat=True)
                student_pool,student_preds = student_model.forward(images,is_feat=True)
            
            feature_map_pairs = []
            feature_map_pairs.append((teacher_pool,student_pool))
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
        kd_loss = self.kd_loss(teacher_preds,student_preds)
        teacher_pool=feature_map_pairs[0][0]
        student_pool=feature_map_pairs[0][1]
        #teacher_pool=teacher_pool.view(teacher_pool.size(0),-1)
        #student_pool=student_pool.view(student_pool.size(0),-1)
        distance_loss=self.distance_loss(student_pool,teacher_pool)
        angle_loss=self.angle_loss(student_pool,teacher_pool)
        lambda_d=self.lambda_['lambda_d']
        lambda_a=self.lambda_['lambda_a']

        return ce_loss + (self.temperature**2)*kd_loss+lambda_d*distance_loss+lambda_a*angle_loss
            
