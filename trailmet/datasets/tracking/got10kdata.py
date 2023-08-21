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
# https://github.com/got-10k/siamfc
from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np
import torch
from collections import namedtuple
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose, CenterCrop, RandomCrop, ToTensor
from PIL import Image, ImageStat, ImageOps

from got10k.datasets import GOT10k


class RandomStretch(object):

    def __init__(self, max_stretch=0.05, interpolation='bilinear'):
        assert interpolation in ['bilinear', 'bicubic']
        self.max_stretch = max_stretch
        self.interpolation = interpolation

    def __call__(self, img):
        scale = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        size = np.round(np.array(img.size, float) * scale).astype(int)
        if self.interpolation == 'bilinear':
            method = Image.BILINEAR
        elif self.interpolation == 'bicubic':
            method = Image.BICUBIC
        return img.resize(tuple(size), method)


class Pairwise(Dataset):

    def __init__(self, seq_dataset, **kargs):
        super(Pairwise, self).__init__()
        self.cfg = self.parse_args(**kargs)

        self.seq_dataset = seq_dataset
        self.indices = np.random.permutation(len(seq_dataset))
        # augmentation for exemplar and instance images
        self.transform_z = Compose([
            RandomStretch(max_stretch=0.05),
            CenterCrop(self.cfg.instance_sz - 8),
            RandomCrop(self.cfg.instance_sz - 2 * 8),
            CenterCrop(self.cfg.exemplar_sz),
            ToTensor(),
        ])
        self.transform_x = Compose([
            RandomStretch(max_stretch=0.05),
            CenterCrop(self.cfg.instance_sz - 8),
            RandomCrop(self.cfg.instance_sz - 2 * 8),
            ToTensor(),
        ])

    def parse_args(self, **kargs):
        # default parameters
        cfg = {
            'pairs_per_seq': 10,
            'max_dist': 100,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
        }

        for key, val in kargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('GenericDict', cfg.keys())(**cfg)

    def __getitem__(self, index):
        index = self.indices[index % len(self.seq_dataset)]
        img_files, anno = self.seq_dataset[index]

        # remove too small objects
        valid = anno[:, 2:].prod(axis=1) >= 10
        img_files = np.array(img_files)[valid]
        anno = anno[valid, :]

        rand_z, rand_x = self._sample_pair(len(img_files))

        exemplar_image = Image.open(img_files[rand_z])
        instance_image = Image.open(img_files[rand_x])
        exemplar_image = self._crop_and_resize(exemplar_image, anno[rand_z])
        instance_image = self._crop_and_resize(instance_image, anno[rand_x])
        exemplar_image = 255.0 * self.transform_z(exemplar_image)
        instance_image = 255.0 * self.transform_x(instance_image)

        return exemplar_image, instance_image

    def __len__(self):
        return self.cfg.pairs_per_seq * len(self.seq_dataset)

    def _sample_pair(self, n):
        assert n > 0
        if n == 1:
            return 0, 0
        elif n == 2:
            return 0, 1
        else:
            max_dist = min(n - 1, self.cfg.max_dist)
            rand_dist = np.random.choice(max_dist) + 1
            rand_z = np.random.choice(n - rand_dist)
            rand_x = rand_z + rand_dist

        return rand_z, rand_x

    def _crop_and_resize(self, image, box):
        # convert box to 0-indexed and center based
        box = np.array(
            [
                box[0] - 1 + (box[2] - 1) / 2,
                box[1] - 1 + (box[3] - 1) / 2,
                box[2],
                box[3],
            ],
            dtype=np.float32,
        )
        center, target_sz = box[:2], box[2:]

        # exemplar and search sizes
        context = self.cfg.context * np.sum(target_sz)
        z_sz = np.sqrt(np.prod(target_sz + context))
        x_sz = z_sz * self.cfg.instance_sz / self.cfg.exemplar_sz

        # convert box to corners (0-indexed)
        size = round(x_sz)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size,
        ))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((-corners[:2], corners[2:] - image.size))
        npad = max(0, int(pads.max()))
        if npad > 0:
            avg_color = ImageStat.Stat(image).mean
            # PIL doesn't support float RGB image
            avg_color = tuple(int(round(c)) for c in avg_color)
            image = ImageOps.expand(image, border=npad, fill=avg_color)

        # crop image patch
        corners = tuple((corners + npad).astype(int))
        patch = image.crop(corners)

        # resize to instance_sz
        out_size = (self.cfg.instance_sz, self.cfg.instance_sz)
        patch = patch.resize(out_size, Image.BILINEAR)

        return patch


class GOT10kDataset:

    def __init__(
        self,
        name=None,
        root=None,
        split_types=None,
        shuffle=True,
        random_seed=None,
    ):
        self.name = name
        self.shuffle = shuffle
        self.dataset_dict = {}

        for item in split_types:
            dataset_type = item
            data = GOT10k(root, subset=dataset_type)
            if item != 'test':
                data = Pairwise(data)
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
            self.dataset_dict['train'])
        self.dataset_dict['info']['val_size'] = len(self.dataset_dict['val'])
        self.dataset_dict['info']['test_size'] = len(self.dataset_dict['test'])
        self.dataset_dict['info']['note'] = ''
        return self.dataset_dict

    def stack_dataset(self):
        """
        Behavior:
            This function stacks the three dataset objects (train, val and test) in a single dictionary together with
            their samplers. For cases where the no validation set is explicitly available, the split is performed here.
        Returns:
            dataset_dict (dict): The keys of the dictionary are "train_datset", "val_dataset"
            and "test_dataset" and the values are object of dataset containing train,
            val and test respectively.
        """

        # defining the samplers
        self.dataset_dict['train_sampler'] = None
        self.dataset_dict['val_sampler'] = None
        self.dataset_dict['test_sampler'] = None

        if self.name == 'got10k':
            self.train_idx, self.valid_idx = range(
                len(self.dataset_dict['train'])), range(
                    len(self.dataset_dict['val']))
            train_sampler = SubsetRandomSampler(self.train_idx)
            valid_sampler = SubsetRandomSampler(self.valid_idx)
            self.dataset_dict['train_sampler'] = train_sampler
            self.dataset_dict['val_sampler'] = valid_sampler

        return self.dataset_dict
