
import os
import glob
import numpy as np
from pickle import Unpickler
import torchvision
from .dataset import BaseDataset



class CIFAR10Dataset(BaseDataset):

    def __init__(self, name=None, root=None,
                transform=None,
                target_transform=None,
                 download=True, 
                 split_types = ['train', 'val', 'test'],
                 val_fraction = 0.2,
                 shuffle=True,
                 random_seed = 42):
        super(CIFAR10Dataset, self).__init__(name=name, root=root,
                 transform=transform,
                 target_transform=target_transform,
                 download=download, 
                 split_types =split_types,
                 val_fraction =val_fraction,
                 shuffle=shuffle,
                 random_seed=random_seed)
    

        dataset = torchvision.datasets.CIFAR10
        self.dataset_dict = {}

        for item in self.split_types:
            dataset_type = item + "_dataset"
            if item == 'val' and not self.val_exists:
                self.dataset_dict[dataset_type] = None
            print(self.transform[item])
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




   