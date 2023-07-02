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
import sys
import torchvision
from .dataset import BaseDataset


class CIFAR10Dataset(BaseDataset):
    """
    `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    The CIFAR-10 dataset consists of 60000 32x32 colour images
    in 10 classes, with 6000 images per class. There are 50000
    training images and 10000 test images.The classes are completely
    mutually exclusive.

    References
    ----------
    Learning Multiple Layers of Features from Tiny Images,
    Alex Krizhevsky, 2008.

    Parameters
    ----------
        name (string): dataset name 'CIFAR10', 'CIFAR100', default=None.
        root (string): Root directory where ``cifar-10-batches-py`` exists or will be saved if download flag is set to
        True (default is None).
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set, default=None.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``. Default=None.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it, default=None.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again, default=True.
        split_types (list): the possible values of this parameter are "train", "test" and "val".
            If the split_type contains "val", then shuffle has to be True, default value is None.
        val_fraction (float): If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to include in the val split.
        shuffle (bool): Whether or not to shuffle the data before splitting into val from train,
            default is True. If shuffle is true, there should be 'val' in split_types.
        random_seed (int): RandomState instance, default=None.
    """

    def __init__(
        self,
        name=None,
        root=None,
        transform=None,
        target_transform=None,
        download=True,
        split_types=None,
        val_fraction=0.2,
        shuffle=True,
        random_seed=None,
    ):
        super(CIFAR10Dataset, self).__init__(
            name=name,
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
            split_types=split_types,
            val_fraction=val_fraction,
            shuffle=shuffle,
            random_seed=random_seed,
        )

        dataset = torchvision.datasets.CIFAR10
        self.dataset_dict = {}

        for item in self.split_types:
            dataset_type = item
            if item == 'val' and not self.val_exists:
                self.dataset_dict[dataset_type] = None
            data = dataset(
                root=self.root,
                train=(item != 'test'),
                transform=self.transform[item],
                target_transform=self.target_transform[item],
                download=self.download,
            )
            self.dataset_dict[dataset_type] = data

    def build_dict_info(self):
        """
        Behavior:
            This function creates info key in the output dictionary. The info key contains details related to the size
            of the training, validation and test datasets. Further, it can be used to define any additional information
            necessary for the user.
        Returns:
            dataset_dict (dict): Updated with info key that contains details related to the data splits
        """
        self.dataset_dict['info'] = {}
        self.dataset_dict['info']['train_size'] = len(
            self.dataset_dict['train_sampler'])
        self.dataset_dict['info']['val_size'] = len(
            self.dataset_dict['val_sampler'])
        self.dataset_dict['info']['test_size'] = len(self.dataset_dict['test'])
        self.dataset_dict['info']['note'] = (
            'Note that we use the CIFAR10 instance of torchvision for train and validation, '
            'due to which the length of these will be displayed as 50000 when len() is invoked.'
            'For accurate details, extract information from the info keyword in this dict '
        )
        return self.dataset_dict


class CIFAR100Dataset(BaseDataset):
    """
    `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    The CIFAR-100 dataset consists of 60000 32x32 colour images
    in 100 classes, with 600 images per class. There are 50000
    training images and 10000 test images. The classes are completely
    mutually exclusive.

    References
    ----------
    Learning Multiple Layers of Features from Tiny Images,
    Alex Krizhevsky, 2008.

    Parameters
    ----------
        name (string): dataset name 'CIFAR10', 'CIFAR100', default=None.
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download
             is set to True, default=None.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set, default=None.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``. Default=None.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it, default=None.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again, default=True.
        split_types (list): the possible values of this parameter are "train", "test" and "val".
            If the split_type contains "val", then suffle has to be True, default value is None.
        val_fraction (float): If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to include in the val split.
        shuffle (bool): Whether or not to shuffle the data before splitting into val from train,
            default is True. If shuffle is true, there should be 'val' in split_types.
        random_seed (int): RandomState instance, default=None.
    """

    def __init__(
        self,
        name=None,
        root=None,
        transform=None,
        target_transform=None,
        download=True,
        split_types=None,
        val_fraction=0.2,
        shuffle=True,
        random_seed=None,
    ):
        super(CIFAR100Dataset, self).__init__(
            name=name,
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
            split_types=split_types,
            val_fraction=val_fraction,
            shuffle=shuffle,
            random_seed=random_seed,
        )

        ## To do: check val_frac is float, else raise error
        ## To do: if shuffle is true, there should be 'val' in train test split
        dataset = torchvision.datasets.CIFAR100
        self.dataset_dict = {}

        for item in self.split_types:
            dataset_type = item
            if item == 'val' and not self.val_exists:
                self.dataset_dict[dataset_type] = None
            data = dataset(
                root=self.root,
                train=(item != 'test'),
                transform=self.transform[item],
                target_transform=self.target_transform[item],
                download=self.download,
            )
            self.dataset_dict[dataset_type] = data

    def build_dict_info(self):
        """
        Behavior:
            This function creates info key in the output dictionary. The info key contains details related to the size
            of the training, validation and test datasets. Further, it can be used to define any additional information
            necessary for the user.
        Returns:
            dataset_dict (dict): Updated with info key that contains details related to the data splits
        """
        self.dataset_dict['info'] = {}
        self.dataset_dict['info']['train_size'] = len(
            self.dataset_dict['train_sampler'])
        self.dataset_dict['info']['val_size'] = len(
            self.dataset_dict['val_sampler'])
        self.dataset_dict['info']['test_size'] = len(self.dataset_dict['test'])
        self.dataset_dict['info']['note'] = (
            'Note that we use the CIFAR100 instance of torchvision for train and validation, '
            'due to which the length of these will be displayed as 50000 when len() is invoked.'
            'For accurate details, extract information from the info keyword in this dict '
        )
        return self.dataset_dict
