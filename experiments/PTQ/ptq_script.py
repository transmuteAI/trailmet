
import os
import torch
from .data import DataManager
from trailmet.models import resnet
from trailmet.models import mobilenet
from trailmet.algorithms.quantize.brecq import BRECQ
from trailmet.algorithms.quantize.bitsplit import BitSplit



SAVED_MODEL_PATH = './pretrained_models/'
DEVICE = 'cuda:0'
SEEDS = [123]
ARCHS = ['ResNet50']
DATASETS = ['Cifar100']
METHODS = ['BitSplit']
PRECISIONS = [0.75, 0.5, 0.25, 0.175, 0.125, 0.1, 0.0625]
ACTIVATIONS = [4,8,16]

method_factory = {
    'BRECQ': BRECQ,
    'BitSplit': BitSplit
}

def load_data(dataset):
    data_object = DataManager()
    trainloader, valloader, testloader = data_object.prepare_data(name=dataset)
    dataloaders = {'train': trainloader, 'calib': valloader, "val": testloader}
    return dataloaders

def load_model(arch, dataset):
    inp = 32 if dataset=='Cifar100' else 64
    out = 100 if dataset=='Cifar100' else 200
    if arch=='ResNet50':
        model = resnet.get_resnet_model('resnet50', out, inp, pretrained=False)
    elif arch=='MobileNetv2':
        model = mobilenet.get_mobilenet_model('mobilenetv2', out)
    else: 
        raise NotImplementedError
    load_path = SAVED_MODEL_PATH+arch+'_'+dataset+'_'+'pretrained.pth'
    if not os.path.isfile(load_path): 
        raise FileNotFoundError
    checkpoint = torch.load(load_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_config(method, seed,):
    kwargs={
        'SEED': seed,
        'GPU_ID': 8

    }
    if method=='BitSplit':
        kwargs['']=3
    return kwargs

for seed in SEEDS:
    for method in METHODS:
        for arch in ARCHS:
            for dataset in DATASETS:
                dataloaders = load_data(dataset)
                for prec in PRECISIONS:
                    for acts in ACTIVATIONS:
                        cnn = load_model(arch, dataset)
                        kwargs = load_config(seed, )
                        qnn = method_factory[method](cnn, dataloaders, **kwargs)
                        qnn.compress_model()
                        del qnn, kwargs, cnn
                del dataloaders

                
