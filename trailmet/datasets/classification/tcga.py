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
from .dataset import BaseDataset
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import gdown
import zipfile
from glob import glob


class TCGA(Dataset):
    """
    `TCGA <https://www.cancer.gov/ccg/research/structural-genomics/tcga/studied-cancers/lung-squamous-cell-carcinoma-study>`_ Dataset.

    References
    ----------

    Parameters
    ----------
        name (string): dataset name 'TCGA', default=None.
        root (string): Root directory where ``splits, images`` exists or will be saved if download flag is set to
        True (default is None).
        mode (string): Mode to run the dataset, options are 'train', 'val', 'test', default='train'.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set, default=None.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``. Default=None.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it, default=None.
        kfold(int): The fold which you want to use. Possible values are 0-9
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
        root=None,
        transform=None,
        target_transform=None,
        kfold=0,
        mode='train',
        download=True,
    ):
        self.labels_dict = {}
        self._kfold = kfold
        self.transform = transform
        self.target_transform = target_transform
        self.class_dict = {'TCGA-LUAD': 0, 'TCGA-LUSC': 1}
        self.split_file = os.path.join(root,
                                       f'tcga_lung/splits_{self._kfold}.csv')
        self.data_path = os.path.join(root, f'512_imgs/fold-{self._kfold}/')
        df = pd.read_csv(self.split_file)

        files = glob(os.path.join(root,
                                  f'512_imgs/fold-{self._kfold}/*/*.png'))

        for filename in files:
            self.labels_dict[filename.split('/')[-1][:-4]] = filename.split(
                '/')[-2]

        self.Train_List, self.Val_List, self.Test_List = (
            df.train.to_list(),
            df.val.dropna().to_list(),
            df.test.dropna().to_list(),
        )

        if mode == 'train':
            self._mode = 'train'
            print('Total Training Sample {}'.format(len(self.Train_List)))
        elif mode == 'val':
            self._mode = 'val'
            print('Total Validating Sample {}'.format(len(self.Val_List)))
        elif mode == 'test':
            self._mode = 'test'
            print('Total test Sample {}'.format(len(self.Test_List)))

    def __len__(self):
        if self._mode == 'train':
            return len(self.Train_List)
        elif self._mode == 'val':
            return len(self.Val_List)
        elif self._mode == 'test':
            return len(self.Test_List)

    def __getitem__(self, idx):
        if self._mode == 'train':
            path = self.Train_List[idx]
            image_path = os.path.join(self.data_path, self.labels_dict[path],
                                      path + '.png')
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img)
            class_name = self.labels_dict[path]
            return img, torch.tensor([self.class_dict[class_name]]).item()
        elif self._mode == 'val':
            path = self.Val_List[idx]
            image_path = os.path.join(self.data_path, self.labels_dict[path],
                                      path + '.png')
            img = Image.open(image_path).convert('RGB')
            if self.target_transform == None:
                img = self.transform(img)
            else:
                img = self.target_transform(img)
            class_name = self.labels_dict[path]
            return img, torch.tensor([self.class_dict[class_name]]).item()
        else:
            path = self.Test_List[idx]
            image_path = os.path.join(self.data_path, self.labels_dict[path],
                                      path + '.png')
            img = Image.open(image_path).convert('RGB')
            if self.target_transform == None:
                img = self.transform(img)
            else:
                img = self.target_transform(img)
            class_name = self.labels_dict[path]
            return img, torch.tensor([self.class_dict[class_name]]).item()


class TCGADataset(BaseDataset):
    """
    `TCGA <https://www.cancer.gov/ccg/research/structural-genomics/tcga/studied-cancers/lung-squamous-cell-carcinoma-study>`_ Dataset.

    References
    ----------


    Parameters
    ----------
        name (string): dataset name 'TCGA', default=None.
        root (string): Root directory where ``splits, images`` exists or will be saved if download flag is set to True (default is None).
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set, default=None.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``. Default=None.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it, default=None.
        kfold (int): The fold which you want to use. Possible values are 0-9
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
        kfold=0,
        download=True,
        split_types=None,
        val_fraction=0.2,
        shuffle=True,
        random_seed=None,
    ):
        super(TCGADataset, self).__init__(
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

        self.val_exists = True
        os.makedirs(root, exist_ok=True)
        final_path = os.path.join(root, 'tcga_512')

        if not os.path.exists(final_path + f'/512_imgs/fold-{kfold}'):
            print(
                f'TCGA dataset is not present in {root}. Downloading the dataset'
            )
            gdown.download(
                id='1bnfg9mq-5NwnKjS7ZlVAySooLCTGAjqb',
                output=f'{root}/tcga_512.zip',
                quiet=False,
            )
            os.makedirs(final_path, exist_ok=True)

            # unzipping the dataset
            with zipfile.ZipFile(f'{root}/tcga_512.zip', 'r') as zip_ref:
                for member in tqdm(zip_ref.infolist(), desc='Extracting '):
                    try:
                        zip_ref.extract(member, final_path)
                    except zipfile.error as e:
                        raise Exception(
                            f'Unable to extract the dataset. Please check the path {root}'
                        )
            print('Removing the zip file')
            os.remove(f'{root}/tcga_512.zip')
            print('Done! downloading the dataset.')

            if not os.path.exists(final_path + f'/512_imgs/fold-{kfold}'):
                raise Exception(
                    f'Unable to download the dataset. Please check the path {root}'
                )
        else:
            print(f'TCGA dataset is present in {final_path}')

        if not os.path.exists(
                os.path.join(root, 'tcga_512', 'tcga_lung',
                             f'splits_{kfold}.csv')):
            print(
                f'TCGA split file is not present in {root}. Downloading the split file'
            )
            gdown.download(
                id='1xBxLz2iToaHaJJovml7Bf4NcJRS_x2CQ',
                output=f'{root}/tcga_512/tcga_lung.zip',
                quiet=False,
            )
            os.makedirs(os.path.join(root, 'tcga_512', 'tcga_lung'),
                        exist_ok=True)

            # unzipping the dataset
            with zipfile.ZipFile(f'{root}/tcga_512/tcga_lung.zip',
                                 'r') as zip_ref:
                for member in tqdm(zip_ref.infolist(), desc='Extracting '):
                    try:
                        zip_ref.extract(member, final_path)
                    except zipfile.error as e:
                        raise Exception(
                            f'Unable to extract the split files. Please check the path {root}'
                        )
            print('Removing the zip file')
            os.remove(f'{root}/tcga_512/tcga_lung.zip')
            print('Done! downloading the split files.')

            if not os.path.exists(
                    os.path.join(root, 'tcga_512', 'tcga_lung',
                                 f'splits_{kfold}.csv')):
                raise Exception(
                    f'Unable to download the split files. Please check the path {root}'
                )
        else:
            print(f'TCGA split files is present in {final_path}')

        dataset = TCGA
        self.dataset_dict = {}

        for item in self.split_types:
            dataset_type = item
            if item == 'val' and not self.val_exists:
                self.dataset_dict[dataset_type] = None
                continue
            data = dataset(
                root=final_path,
                mode=item,
                kfold=kfold,
                transform=self.transform[item],
                target_transform=self.target_transform[item],
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
            self.dataset_dict['train'])
        self.dataset_dict['info']['val_size'] = len(self.dataset_dict['val'])
        self.dataset_dict['info']['test_size'] = len(self.dataset_dict['test'])
        self.dataset_dict['info']['note'] = ''
        return self.dataset_dict
