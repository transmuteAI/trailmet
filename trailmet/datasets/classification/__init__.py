import os
from .cifar import CIFAR10Dataset


class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'CIFAR10', 'CIFAR100', 'ImageNet',
            root: root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        assert 'root' in kwargs, "should provide dataset root"
        if 'CIFAR10' == name:
            obj_dfactory = CIFAR10Dataset(**kwargs)
            dataset =  obj_dfactory.stack_dataset()
        elif 'CIFAR100' == name:
            dataset = CIFAR100Dataset(**kwargs)
        elif 'ImageNet' == name:
            dataset = ImageNetDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset