import os
import sys

path = "../../"
sys.path.append(path)

import torch
from torchvision import transforms as transforms
from torchvision import datasets as datasets
torch.cuda.set_device(7)

def build_imagenet_data(data_path: str = '', input_size: int = 224, batch_size: int = 64, workers: int = 4,
                        dist_sample: bool = False):
    print('==> Using Imagenet Dataset')

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #torchvision.set_image_backend('accimage')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]))

    if dist_sample:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)
    return train_loader, val_loader

    # load data
dataloaders = {'train':[], 'val':[]}
dataloaders['train'], dataloaders['val'] = build_imagenet_data(data_path='/workspace/code/Akash/ImageNet')

# import libraries
from trailmet.models.resnet import *
from trailmet.algorithms.quantize.lapq import LAPQ
from trailmet.algorithms.quantize.quantize import BaseQuantization

# load model
cnn = get_resnet_model('resnet50', 1000, 224, pretrained=False)

# test full precision model
bq = BaseQuantization()
bq.test(model=cnn, dataloader=dataloaders['val'], device=torch.device('cuda:7'))

# quantize model
kwargs = {
    # 'ACT_QUANT':True, 
    'NUM_SAMPLES':1024, 
    'ITERS_W':2000, 
    'ITERS_A':2000, 
    'cal_batch_size':256,
    'cal_set_size':512,
    'GPU_ID':7,
    'pretrained':True,
    'min_method': 'Powell',
    'maxiter' : 2,
    'bit_act' : 4,
    'bit_weights' : 4
    }
lapq = LAPQ(cnn, dataloaders, **kwargs)
lapq.compress_model()

# increase iterations from 2000 to 20000 to achieve full optimization potential.

torch.cuda.empty_cache()
