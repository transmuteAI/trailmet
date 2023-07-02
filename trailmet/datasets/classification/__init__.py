#!/usr/bin/env python

# MIT License
#
# Copyright (c) 2023 Transmute AI Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# importing the required packages
import os
from .cifar import CIFAR10Dataset, CIFAR100Dataset
from .imagenet import ImageNetDataset
from .chest import ChestDataset
from .tcga import TCGADataset


class DatasetFactory(object):
    """This class forms the generic wrapper for the different dataset classes.

    The module includes utilities to load datasets, including methods to load
    and fetch popular reference datasets.
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
        assert 'name' in kwargs, 'should provide dataset name'
        name = kwargs['name']
        assert 'root' in kwargs, 'should provide dataset root'
        if 'CIFAR10' == name:
            obj_dfactory = CIFAR10Dataset(**kwargs)
        elif 'CIFAR100' == name:
            obj_dfactory = CIFAR100Dataset(**kwargs)
        elif 'ImageNet' == name:
            obj_dfactory = ImageNetDataset(**kwargs)
        elif 'CHEST' == name:
            obj_dfactory = ChestDataset(**kwargs)
        elif 'TCGA' == name:
            obj_dfactory = TCGADataset(**kwargs)
        else:
            raise Exception(f"unknown dataset{kwargs['name']}")
        dataset = obj_dfactory.stack_dataset()
        dataset = obj_dfactory.build_dict_info()

        return dataset
