
import os
import torchvision
from ..dataset import Dataset

class CIFAR10Dataset(Dataset):

    def __init__(self, name=None, root=None,
                 train=True, transform=None,
                 target_transform=None,
                 download=True):
        super(CIFAR10Dataset, self).__init__(name=name, root=root,
                 train=train, transform=transform,
                 target_transform=target_transform,
                 download=download)
        dataset = torchvision.datasets.CIFAR10
        if self.train == True:
            trainset = dataset(root=self.root, 
                        train=self.train,
                        transform=self.transform,
                        target_transform = self.target_transform,
                        download=self.download)
            
        else:
            testset = dataset(root=self.root, 
                        train=self.train,
                        transform=self.transform,
                        target_transform = self.target_transform,
                        download=self.download)
            


class CIFAR100Dataset(Dataset):

    def __init__(self, name=None, root=None,
                 train=True, transform=None,
                 target_transform=None,
                 download=True):
        super(CIFAR100Dataset, self).__init__(name=None, root=None,
                 train=True, transform=None,
                 target_transform=None,
                 download=True)
        dataset = torchvision.datasets.CIFAR100
        if self.train == True:
            trainset = dataset(root=self.root, 
                        train=self.train,
                        transform=self.transform,
                        target_transform = self.target_transform,
                        download=self.download)
            return trainset
        else:
            testset = dataset(root=self.root, 
                        train=self.train,
                        transform=self.transform,
                        target_transform = self.target_transform,
                        download=self.download)
            return testset


