#!/usr/bin/env python

# importing the required packages
import os
from .cifar import CIFAR10Dataset, CIFAR100Dataset
from .imagenet import ImageNetDataset
from .chest import ChestDataset


class DatasetFactory(object):
    """
    This class forms the generic wrapper for the different dataset classes. The module includes utilities to load datasets,
    including methods to load and fetch popular reference datasets.
    """

    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name(string): dataset name 'CIFAR10', 'CIFAR100', 'ImageNet', 'CHEST',
            root(string):  Root directory of dataset where directory
                   cifar-10-batches-py exists or will be saved
                   to if download is set to True.
        Return:
            dataset(tuple): dataset
        """
        assert "name" in kwargs, "should provide dataset name"
        name = kwargs["name"]
        assert "root" in kwargs, "should provide dataset root"
        if "CIFAR10" == name:
            obj_dfactory = CIFAR10Dataset(**kwargs)
        elif "CIFAR100" == name:
            obj_dfactory = CIFAR100Dataset(**kwargs)
        elif "ImageNet" == name:
            obj_dfactory = ImageNetDataset(**kwargs)
        elif "CHEST" == name:
            obj_dfactory = ChestDataset(**kwargs)
        else:
            raise Exception(f"unknown dataset{kwargs['name']}")
        dataset = obj_dfactory.stack_dataset()
        dataset = obj_dfactory.build_dict_info()

        return dataset
