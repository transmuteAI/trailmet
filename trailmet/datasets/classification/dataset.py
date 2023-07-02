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
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataset:
    """BaseDataset class is to be inherited by all dataset classes. All the
    generic attributes and methods will be a part of this class.

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
        random_seed=42,
    ):
        self.name = name
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.split_types = split_types
        self.val_fraction = val_fraction
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.val_exists = False
        self.dataset_dict = {}

        # performing QC of the input parameters
        try:
            self.val_fraction = float(
                self.val_fraction)  # check for float type
        except ValueError:
            raise ValueError('val_fraction needs to be of float type')

        assert (self.val_fraction >= 0.0) and (
            self.val_fraction
            < 1.0), 'val_fraction should be in range [0.0, 1.0]'

        # check if split_types contain nothing beyond train, val and test
        assert (
            len(set(self.split_types) - set(['train', 'val', 'test'])) == 0
        ), 'Only train, val and test permitted as inputs in split_types'

        if self.shuffle:
            # ensure that val exists in split_types
            assert (
                'val' in self.split_types
            ), 'Required param val missing in the defined split types (split_type)'

    def stack_dataset(self):
        """
        Behavior:
            This function stacks the three dataset objects (train, val and test) in a single dictionary together with
            their samplers. For cases where the no validation set is explicitly available, the split is performed here.
        Returns:
            dataset_dict (dict): The keys of the dictionary are "train_datset", "val_dataset"
            and "test_dataset" and the values are object of pytorch CIFER10 dataset containing train,
            val and test respectively.
        """

        # defining the samplers
        self.dataset_dict['train_sampler'] = None
        self.dataset_dict['val_sampler'] = None
        self.dataset_dict['test_sampler'] = None

        if self.name in ['CIFAR10', 'CIFAR100', 'CHEST']:
            num_train = len(self.dataset_dict['train'])
            indices = list(range(num_train))
            split = int(np.floor(self.val_fraction * num_train))

            if self.shuffle:
                np.random.seed(self.random_seed)
                np.random.shuffle(indices)

            self.train_idx, self.valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(self.train_idx)
            valid_sampler = SubsetRandomSampler(self.valid_idx)
            self.dataset_dict['train_sampler'] = train_sampler
            self.dataset_dict['val_sampler'] = valid_sampler
        elif self.name == 'TCGA':
            self.train_idx, self.valid_idx = range(
                len(self.dataset_dict['train'])), range(
                    len(self.dataset_dict['val']))
            train_sampler = SubsetRandomSampler(self.train_idx)
            valid_sampler = SubsetRandomSampler(self.valid_idx)
            self.dataset_dict['train_sampler'] = train_sampler
            self.dataset_dict['val_sampler'] = valid_sampler

        return self.dataset_dict
