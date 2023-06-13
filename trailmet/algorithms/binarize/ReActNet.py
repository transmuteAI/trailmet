from trailmet.models.resnet_react import ResNet, BasicBlock1, BasicBlock2
from trailmet.models.mobilenet_1 import reactnet_1
from trailmet.models.mobilenet_2 import reactnet_2
import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from trailmet.algorithms.binarize.utils import *
sys.path.append('/workspace/code/kushagrabhushan/TrAIL/trailmet/trailmet/algorithms/binarize')
sys.path.append("../../../")
from utils import *
from trailmet.utils import *


class ReActNet():
    def __init__(self, teacher, model_name, dataloaders, num_fp, **kwargs):
        self.teacher = teacher
        self.model_name = model_name
        self.num_fp=num_fp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloaders=dataloaders
        self.kwargs=kwargs

        self.dataset = self.kwargs['GENERAL'].get('DATASET', 'c100')
        self.num_classes = self.kwargs['GENERAL'].get('num_classes',100)
        self.insize = self.kwargs['GENERAL'].get('insize',32)
        self.batch_size1 = self.kwargs['ReActNet1_ARGS'].get('batch_size',128)
        self.epochs1 = self.kwargs['ReActNet1_ARGS'].get('epochs',128)
        self.learning_rate1 = self.kwargs['ReActNet1_ARGS'].get('learning_rate',2.5e-3)
        self.momentum1 = self.kwargs['ReActNet1_ARGS'].get('momentum',0.9)
        self.weight_decay1 = self.kwargs['ReActNet1_ARGS'].get('weight_decay',1e-5)
        self.label_smooth1 = self.kwargs['ReActNet1_ARGS'].get('label_smooth',0.1)
        self.save1 = self.kwargs['ReActNet1_ARGS'].get('save','')

        self.batch_size2 = self.kwargs['ReActNet2_ARGS'].get('batch_size',128)
        self.epochs2 = self.kwargs['ReActNet2_ARGS'].get('epochs',128)
        self.learning_rate2 = self.kwargs['ReActNet2_ARGS'].get('learning_rate',2.5e-3)
        self.momentum2 = self.kwargs['ReActNet2_ARGS'].get('momentum',0.9)
        self.weight_decay2 = self.kwargs['ReActNet2_ARGS'].get('weight_decay',0)
        self.label_smooth2 = self.kwargs['ReActNet2_ARGS'].get('label_smooth',0.1)
        self.save2 = self.kwargs['ReActNet2_ARGS'].get('save','')
    
    def prepare_logs(self):
        if not os.path.exists('log'):
                print('Creating Logging Directory...')
                os.mkdir('log')
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join('log/log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
    
    def make_model_actbin(self):
        if self.model_name == 'resnet50':
            new_model=ResNet(BasicBlock1, [3, 4, 6, 3], self.num_fp, width=1, num_classes=self.num_classes, insize=self.insize)
        elif self.model_name == 'mobilenetv2':
            new_model = reactnet_1(num_classes=self.num_classes)
        else:
            print("Model Not Avaliable")
        return new_model

    def make_model_fullbin(self):
        if self.model_name == 'resnet50':
            new_model=ResNet(BasicBlock2, [3, 4, 6, 3], self.num_fp, width=1, num_classes=self.num_classes, insize=self.insize)
        elif self.model_name == 'mobilenetv2':
            new_model = reactnet_2(num_classes=self.num_classes)
        else:
            print("Model Not Avaliable")
        return new_model

    def train_one_epoch(self,model,teacher,scheduler,criterion,optimizer):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        
        train_loader = self.dataloaders['train']
        
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, losses, top1, top5],
            prefix='Train: ')

        scheduler.step()
        end = time.time()
        
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('learning_rate:', cur_lr)
        
        for i, (images, target) in enumerate(train_loader):
            images = images.cuda()
            target = target.cuda()

            # compute outputy
            logits_student = model(images)
            logits_teacher = teacher(images)
            loss = criterion(logits_student, logits_teacher)
            prec1, prec5 = accuracy(logits_student, target, topk1=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)   #accumulated loss
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)

            
        return losses.avg, top1.avg, top5.avg

    def base_train(self,model,teacher,epochs,criterion,scheduler,optimizer,save):
        model.train()
        teacher = teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        epoch=0
        best_top1_acc=0
        logging.info("epoch, train acc, train loss, val acc, val loss")
        while(epoch<epochs):
            print("EPOCH-{}\n".format(epoch))
            train_obj, train_top1_acc,  train_top5_acc = self.train_one_epoch(model, teacher, scheduler, criterion, optimizer)
            v_loss, acc, acc5 = self.test(model, self.dataloaders['val'], nn.CrossEntropyLoss())
            
            is_best=False
            if(acc>best_top1_acc):
                best_top1_acc=acc
                is_best=True
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_top1_acc': best_top1_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, save, self.dataset)

            print("Top-1 Train Accuracy-{0}\nTop-5 Train Accuracy-{1}\nValidation Accuracy-{2}".format(train_top1_acc,train_top5_acc,acc))
            logging.info("{}, {}, {}, {}, {}".format(epoch, train_top1_acc, train_obj, acc.item(), v_loss))
            epoch+=1
        
        return model


    def train_actbin(self):
        print("Step-1 Training with activations binarized for {} epochs\n".format(self.epochs1))
        logging.info("Step-1 Training with activations binarized for {} epochs\n".format(self.epochs1))
        model = self.make_model_actbin()
        model = model.to(self.device)
        teacher = self.teacher.to(self.device)

        all_parameters = model.parameters()
        weight_parameters = []
        for pname, p in model.named_parameters():
            if p.ndimension() == 4 or 'conv' in pname:
                weight_parameters.append(p)
        weight_parameters_id = list(map(id, weight_parameters))
        other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))


        criterion = DistributionLoss()
        optimizer = torch.optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : self.weight_decay1}],
            lr=self.learning_rate1,)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/self.epochs1), last_epoch=-1)
        
        model = self.base_train(model,teacher,self.epochs1,criterion,scheduler,optimizer,self.save1)
        return model
    
    
    def train_fullbin(self,model):
        print("Step-2 Training with both activations and weights binarized for {} epochs".format(self.epochs2))
        logging.info("Step-2 Training with both activations and weights binarized for {} epochs".format(self.epochs2))
        teacher = self.teacher.to(self.device)

        all_parameters = model.parameters()
        weight_parameters = []
        for pname, p in model.named_parameters():
            if p.ndimension() == 4 or 'conv' in pname:
                weight_parameters.append(p)
        weight_parameters_id = list(map(id, weight_parameters))
        other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))


        criterion = DistributionLoss()
        optimizer = torch.optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : self.weight_decay2}],
            lr=self.learning_rate2,)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/self.epochs2), last_epoch=-1)
        
        model = self.base_train(model,teacher,self.epochs2,criterion,scheduler,optimizer,self.save2)
        return model

    def test(self, model, test_loader, criterion):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(test_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        # switch to evaluation mode
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(test_loader):
                images = images.to(device=self.device)
                target = target.to(device=self.device)

                # compute output
                logits = model(images)
                loss = criterion(logits, target)

                # measure accuracy and record loss
                pred1, pred5 = accuracy(logits, target, topk1=(1, 5))
                n = images.size(0)
                losses.update(loss.item(), n)
                top1.update(pred1[0], n)
                top5.update(pred5[0], n)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                progress.display(i)

        return losses.avg, top1.avg, top5.avg
   
    def compress_model(self):
        self.prepare_logs()
        model_actbin = self.train_actbin()
        model_fullbin = self.make_model_fullbin()
        model_fullbin = model_fullbin.to(self.device)
        model_fullbin.load_state_dict(model_actbin.state_dict(),strict=False)
        fin_output = self.train_fullbin(model_fullbin)

        return fin_output
