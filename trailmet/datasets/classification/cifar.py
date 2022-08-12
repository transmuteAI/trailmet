"""Cifer10, Cifer100 dataset.
The original database is available from StatLib
    <https://www.cs.toronto.edu/~kriz/cifar.html>
The CIFAR-10 dataset consists of 60000 32x32 colour images
in 10 classes, with 6000 images per class. There are 50000
training images and 10000 test images.The classes are completely
mutually exclusive.

This dataset is just like the CIFAR-10, except it has 100 classes
containing 600 images each. There are 500 training images and 100
testing images per class.
References
----------
Learning Multiple Layers of Features from Tiny Images,
Alex Krizhevsky, 2008.
"""
import torchvision
from .dataset import BaseDataset



class CIFAR10Dataset(BaseDataset):
    """
    `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

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
    def __init__(self, name=None, root=None,
                transform=None,
                target_transform=None,
                 download=True,
                 split_types = None,
                 val_fraction = 0.2,
                 shuffle=True,
                 random_seed = None):
        super(CIFAR10Dataset, self).__init__(name=name, root=root,
                 transform=transform,
                 target_transform=target_transform,
                 download=download,
                 split_types =split_types,
                 val_fraction =val_fraction,
                 shuffle=shuffle,
                 random_seed=random_seed)

        ## To do: chcek val_frac is float, else raise error
        ## To do: if shuffle is true, there should be 'val' in train test split
        dataset = torchvision.datasets.CIFAR10
        self.dataset_dict = {}

        for item in self.split_types:
            dataset_type = item + "_dataset"
            if item == 'val' and not self.val_exists:
                self.dataset_dict[dataset_type] = None
            data = dataset(root=self.root,
                        train=(item !="test"),
                        transform=self.transform[item],
                        target_transform = self.target_transform[item],
                        download=self.download)
            self.dataset_dict[dataset_type] = data


class CIFAR100Dataset(BaseDataset):
    """
    `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

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
    def __init__(self, name=None, root=None,
                transform=None,
                target_transform=None,
                 download=True,
                 split_types = None,
                 val_fraction = 0.2,
                 shuffle=True,
                 random_seed = None):
        super(CIFAR100Dataset, self).__init__(name=name, root=root,
                 transform=transform,
                 target_transform=target_transform,
                 download=download,
                 split_types =split_types,
                 val_fraction =val_fraction,
                 shuffle=shuffle,
                 random_seed=random_seed)

        ## To do: chcek val_frac is float, else raise error
        ## To do: if shuffle is true, there should be 'val' in train test split
        dataset = torchvision.datasets.CIFAR100
        self.dataset_dict = {}

        for item in self.split_types:
            dataset_type = item + "_dataset"
            if item == 'val' and not self.val_exists:
                self.dataset_dict[dataset_type] = None
            data = dataset(root=self.root,
                        train=(item !="test"),
                        transform=self.transform[item],
                        target_transform = self.target_transform[item],
                        download=self.download)
            self.dataset_dict[dataset_type] = data















#         else:
#             testset = dataset(root=self.root,
#                         train=self.train,
#                         transform=self.transform,
#                         target_transform = self.target_transform,
#                         download=self.download)



# class CIFAR100Dataset(Dataset):

#     def __init__(self, name=None, root=None,
#                  train=True, transform=None,
#                  target_transform=None,
#                  download=True,
#                  split_types = ['train', 'val', 'test'],
#                  val_fraction = 0.2,
#                  random_state = 42):
#         super(CIFAR100Dataset, self).__init__(name=None, root=None,
#                  train=True, transform=None,
#                  target_transform=None,
#                  download=True, split_types = ['train', 'val', 'test'],
#                  val_fraction = 0.2,
#                  random_state = 42)
#         dataset = torchvision.datasets.CIFAR100
#         if self.train == True:
#             trainset = dataset(root=self.root,
#                         train=self.train,
#                         transform=self.transform,
#                         target_transform = self.target_transform,
#                         download=self.download)
#             return trainset
#         else:
#             testset = dataset(root=self.root,
#                         train=self.train,
#                         transform=self.transform,
#                         target_transform = self.target_transform,
#                         download=self.download)
#             return testset


# class Create_CIFAR10Dataset(Create_Dataset):
#     def __init__(self, name=None, root=None,
#                  train=True, data=None,
#                  labels=None, transform=None,
#                  target_transform=None,
#                  download=True,
#                  split_types = ['train', 'val', 'test'],
#                  val_fraction = 0.2,
#                  random_state = 42):
#         super(CIFAR10Dataset, self).__init__(name=name, root=root,
#                  train=train, transform=transform,
#                  target_transform=target_transform,
#                  download=download, split_types = ['train', 'val', 'test'],
#                  val_fraction = 0.2,
#                  random_state = 42)
#         return self
#     # download the data
#     dataset = torchvision.datasets.CIFAR10
#     if self.train == True:
#         trainset = dataset(root=self.root,
#                     train=self.train,
#                     transform=self.transform,
#                     target_transform = self.target_transform,
#                     download=self.download)
#     train_file = glob.glob(self.root + "/*")
#     self.data, self.labels = self.load_train_data(train_file)
#     self.train_data, self.valid_data, self.train_labels, self.valid_labels = self.get_train_val_split(self.data, self.labels,
#                                                             test_size=self.val_fraction,
#                                                             random_state=self.random_state)
#     else:
#         testset = dataset(root=self.root,
#                     train=False,
#                     transform=self.transform,
#                     target_transform = self.target_transform,
#                     download=self.download)
#         test_file = os.path.join(BASE_DIR, f"test_batch")
#         dict = self.unpickle(test_file)
#         self.test_data = dict[b'data']
#         self.test_labels = dict[b'labels']

#      def load_train_data(self, root):
#         root = self.root
#         data = np.empty((0, 3072))
#         labels = []
#         for i in range(1,6):
#             path = os.path.join(root, f"data_batch_{i}")
#             dict = self.unpickle(path)
#             data = np.vstack((data, dict[b'data']))
#             labels.extend(dict[b'labels'])
#         return data, labels




