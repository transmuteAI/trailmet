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
import torchvision
from .dataset import BaseDataset


class ImageNetDataset(BaseDataset):
    """
    ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Parameters
    ----------
        name (string): dataset name 'ImageNet', default=None.
        root (string): Root directory of dataset or will be saved to if download
             is set to True, default=None.
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
        super(ImageNetDataset, self).__init__(
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
        dataset = torchvision.datasets.ImageNet
        self.dataset_dict = {}
        for item in self.split_types:
            dataset_type = item + '_dataset'
            if item == 'test':
                data = dataset(
                    root=self.root,
                    split='val',
                    transform=self.transform[item],
                    target_transform=self.target_transform[item],
                )
            else:
                data = dataset(
                    root=self.root,
                    split='train',
                    transform=self.transform[item],
                    target_transform=self.target_transform[item],
                )
            self.dataset_dict[dataset_type] = data
