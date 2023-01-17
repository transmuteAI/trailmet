# importing the required packages
from .dataset import BaseDataset 
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset



class Chest(Dataset):
    """
    `CheXpert <https://stanfordmlgroup.github.io/competitions/chexpert/>`_ Dataset.
    The CheXpert dataset contains 224,316 chest radiographs of 65,240 patients with both frontal and lateral views available. The task is to do automated chest x-ray interpretation, featuring uncertainty labels and radiologist-labeled reference standard evaluation sets.

    References
    ----------
    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison,
    Irvin et al, 2019.

    Parameters
    ----------
        name (string): dataset name 'CHEST', default=None.
        root (string): Root directory where ``train, train.csv`` exists or will be saved if download flag is set to
        True (default is None).
        mode (string): Mode to run the dataset, options are 'train', 'val', 'test', 'heatmap', default='train'.
        subname (string): Subname of the dataset, default='atelectasis'. options are 'atelectasis', 'edema', 'cardiomegaly', 'effusion'
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
    def __init__(self, root=None,
                 subname='atelectasis',
                 transform=None,
                 target_transform=None,
                 mode='train',
                 download=True,
                ):

        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.transform = transform
        self.dict = [{'1.0': 1, '': 0, '0.0': 0, '-1.0': 2}, # need to check this with deepak for multiclass setting.
                     {'1.0': 1, '': 0, '0.0': 0, '-1.0': 1}, ]

        if mode == "train":
            label_path = os.path.join(root, "train.csv")  
        elif mode == "val":
            label_path = os.path.join(root, "valid.csv")
        elif mode == "test":
            label_path = os.path.join(root, "valid.csv") # need to change to test.csv after final check
        
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            for i, value in enumerate(header):
                if subname == value.lower() or (subname == "effusion" and value.split(" ")[-1].lower() == subname):
                    subname_index = i

            for line in tqdm(f):
                fields = line.strip('\n').split(',')
                image_path = os.path.join(root, "/".join(fields[0].split("/")[1:]))
                if subname in ["atelectasis", "edema"]:
                    value = fields[subname_index]
                    self._labels.append(self.dict[1].get(value))
                elif subname in ["cardiomegaly", "effusion"]:
                    value = fields[subname_index]
                    self._labels.append(self.dict[0].get(value))

                self._image_paths.append(image_path)
                assert os.path.exists(image_path), image_path
        self._num_image = len(self._image_paths)        

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = Image.open(self._image_paths[idx]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor([self._labels[idx]]).float()

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'val':
            return (image, label)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, label)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))

class ChestDataset(BaseDataset):
    def __init__(self, name=None, root=None,
                 transform=None,
                 subname='atelectasis',
                 target_transform=None,
                 download=True,
                 split_types = None,
                 val_fraction = 0.2,
                 shuffle=True,
                 random_seed = None):
        super(ChestDataset, self).__init__(name=name, root=root,
                 transform=transform,
                 target_transform=target_transform,
                 download=download,
                 split_types =split_types,
                 val_fraction =val_fraction,
                 shuffle=shuffle,
                 random_seed=random_seed)

        dataset = Chest
        self.dataset_dict = {}

        for item in self.split_types:
            dataset_type = item
            if item == 'val' and not self.val_exists:
                self.dataset_dict[dataset_type] = None
            data = dataset(root=root,
                        subname=subname,
                        mode=item,
                        transform=self.transform[item],
                        target_transform =self.target_transform[item],
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
        self.dataset_dict['info']['train_size'] = len(self.dataset_dict['train'])
        self.dataset_dict['info']['val_size'] = len(self.dataset_dict['val'])
        self.dataset_dict['info']['test_size'] = len(self.dataset_dict['test'])
        self.dataset_dict['info']['note'] = ''
        return self.dataset_dict


