"""
The :mod:`trailmet.datasets` module includes utilities to load datasets,
including methods to load and fetch popular reference datasets.
"""
import os
#from .cifar import CIFAR10Dataset, CIFAR100Dataset, ImageNetDataset
from .cifar import CIFAR10Dataset

class DatasetFactory(object):
    """
    docstring to be written
    """
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name(string): dataset name 'CIFAR10', 'CIFAR100', 'ImageNet',
            root(string):  Root directory of dataset where directory
                   cifar-10-batches-py exists or will be saved
                   to if download is set to True.
        Return:
            dataset(tuple): dataset
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
            raise Exception("unknown dataset{}".format(kwargs['name']))
        return dataset