
import os
import torchvision
from .dataset import BaseDataset


class ImageNetDataset(BaseDataset):

    def __init__(self, name=None, root=None,
                 train=True, transform=None,
                 targer_transform=None,
                 download=True, split=None):
        super(ImageNetDataset, self).__init__(name=None, root=None,
                 train=True, transform=None,
                 targer_transform=None,
                 download=True)
        self.split = split
        dataset = torchvision.datasets.ImageNet
        if self.train == True:
            trainset = dataset(root=self.root, 
                        transform=self.transform,
                        target_transform = self.target_transform,
                        download=self.download,
                        split = "train")
            return trainset
        else:
            testset = dataset(root=self.root, 
                        transform=self.transform,
                        target_transform = self.target_transform,
                        download=self.download,
                        split = "val")
            return testset

