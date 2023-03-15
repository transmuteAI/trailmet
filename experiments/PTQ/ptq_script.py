
import os
import argparse
import torch
import pandas as pd
from .data import DataManager
from trailmet.models import resnet
from trailmet.models import mobilenet
from trailmet.algorithms.quantize.brecq import BRECQ
from trailmet.algorithms.quantize.bitsplit import BitSplit
from trailmet.algorithms.algorithms import BaseAlgorithm


parser = argparse.ArgumentParser(description='PyTorch Model Quantization')
parser.add_argument('-p', '--model_path', default='./pretrained_models/', type=str, 
    help='pretrained model weights path')
parser.add_argument('-g', '--gpu_id', default=0, type=int, 
    help='GPU id to use')
parser.add_argument('-s', '--seeds', default=[123,], type=list,
    help='list of random seed values for running experiments')
parser.add_argument('-c', '--config_file', default='./config.yaml', type=str,
    help='configurations file')
parser.add_argument('-a', '--arch', default='config', type=str, 
    choices=['res50', 'mobv2'], help='model arch, use config for multiple')
parser.add_argument('-d', '--dataset', default='config', type=str)
parser.add_argument('-m', '--method')
parser.add_argument('-l', '--load_saved', default=False, action='store_true',
    help='load precomputed scales')

SAVED_MODEL_PATH = './pretrained_models/'
LOAD_SAVED = True
DEVICE = 'cuda:0'
SEEDS = [123,]
ARCHS = [
    'ResNet50',
    # 'MobileNetV2',
    ]
DATASETS = [  # name, im_size, num_classes, batch_size, code
    ['CIFAR100', 32, 100, 128, 'c100'],
    # ['TinyImagenet', 64, 200, 64, 'tin'],
    ]
METHODS = [
    'BitSplit',
    # 'BRECQ',
    # 'LAPQ',
    ]
COMPRESSION_RATIOS = [0.75, 0.5, 0.25, 0.175, 0.125, 0.1, 0.065]
ACTIVATION_BITS = [32, 16, 8, 4]

def load_data(dataset):
    data_object = DataManager()
    trainloader, valloader, testloader = data_object.prepare_data(
        name=dataset[0], batch_size=dataset[3])
    dataloaders = {'train': trainloader, 'val': valloader, "test": testloader}
    return dataloaders

def load_model(arch, dataset):
    inp = dataset[1]
    out = dataset[2]
    if arch=='ResNet50':
        model = resnet.get_resnet_model('resnet50', out, inp, pretrained=False)
    elif arch=='MobileNetv2':
        model = mobilenet.get_mobilenet_model('mobilenetv2', out)
    else: 
        raise NotImplementedError
    load_path = SAVED_MODEL_PATH+arch.lower()+'_'+dataset[4]+'_'+'pretrained.pth'
    if not os.path.isfile(load_path): 
        raise FileNotFoundError
    checkpoint = torch.load(load_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    return model

bitsplit_mobv2_config_map = {
    0.065 : {'W_BITS': 2, 'HEAD_STEM_PRECISION': 3},
    0.1   : {'W_BITS': 3, 'HEAD_STEM_PRECISION': 4},
    0.125 : {'W_BITS': 4, 'HEAD_STEM_PRECISION': 4},
    0.175 : {'PREC_CONFIG': [8,8,3,3,8,8,8]},
    0.25  : {'W_BITS': 8, 'HEAD_STEM_PRECISION': 8},
    0.5   : {'W_BITS':16, 'HEAD_STEM_PRECISION':16},
    0.75  : {'PREC_CONFIG': [32,32,16,16,32,32,32]},
}

def load_config(method, arch, dataset, seed, prec, acts):
    kwargs={
        'SEED': seed,
        'GPU_ID': int(DEVICE[-1]),
        'ARCH': arch,
        'DATASET': dataset[0],
        'A_BITS': acts,
        'ACT_QUANT': (acts!=32)
    }
    if method=='BitSplit':
        kwargs.update(bitsplit_mobv2_config_map[prec])
        kwargs['SAVE_PATH'] = './bitsplit_scales/'
    return kwargs

df  = pd.DataFrame()
for seed in SEEDS:
    for method in METHODS:
        for arch in ARCHS:
            for dataset in DATASETS:
                dataloaders = load_data(dataset)
                cnn = load_model(arch, dataset)
                for prec in COMPRESSION_RATIOS:
                    started = False
                    for acts in ACTIVATION_BITS:
                        kwargs = load_config(method, arch, dataset, seed, prec, acts)
                        kwargs['LOAD_WEIGHT_SCALES'] = (LOAD_SAVED or started)
                        kwargs['LOAD_ACT_SCALES'] = LOAD_SAVED
                        qnn = eval(f'method({cnn}, {dataloaders}, **{kwargs})')
                        qnn.compress_model()
                        top1, top5 = BaseAlgorithm().test(
                            model=qnn.qmodel, dataloader=dataloaders['test'], 
                            device=torch.device(DEVICE)
                        )
                        started = True
                        del qnn, kwargs
                del dataloaders

            