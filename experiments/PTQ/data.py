
import os, glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.utils import download_and_extract_archive
from sklearn.model_selection import train_test_split


class DataManager:
    def __init__(self):
        self.batch_size = 128
        self.workers = 2
        self.valid_size = 0.1
        self.num_train = 0

    def prepare_data(self, name):
        print('... Preparing data ...')

        if name in ['Cifar100', 'cifar100', 'c100']:
            norm_mean = [0.49139968, 0.48215827, 0.44653124]
            norm_std = [0.24703233, 0.24348505, 0.26158768]
            norm_transform = transforms.Normalize(norm_mean, norm_std)
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm_transform
            ])
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                norm_transform
            ])
            trainset = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
            valset = CIFAR100(root='./data', train=True, download=True, transform=val_transform)
            testset = CIFAR100(root='./data', train=False, download=True, transform=val_transform)
                                                
        elif name in ['TinyImagenet', 'tiny_imagenet', 'tin']:
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
            norm_transform = transforms.Normalize(norm_mean, norm_std)
            train_transform = transforms.Compose([
                transforms.RandomAffine(degrees=20.0, scale=(0.8, 1.2), shear=20.0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm_transform,
            ])
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                norm_transform
            ])
            trainset = TinyImageNet('./data', train=True, transform=train_transform)
            valset = TinyImageNet('./data', train=True, transform=val_transform)
            testset = TinyImageNet('./data', train=False, transform=val_transform)

        else: raise NotImplementedError

        self.num_train = len(trainset)
        train_idx, val_idx = self.get_split()
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(trainset, self.batch_size, num_workers=self.workers, sampler=train_sampler, pin_memory=True)
        val_loader = DataLoader(valset, self.batch_size, num_workers=self.workers, sampler=val_sampler, pin_memory=True)
        test_loader = DataLoader(testset, self.batch_size, num_workers=self.workers, shuffle=False, pin_memory=False)
        
        return train_loader, val_loader, test_loader


    def get_split(self):
        if(os.path.exists(f'data_splits/{self.dataset_name}_train_idx.npy') and os.path.exists(f'data_splits/{self.dataset_name}_valid_idx.npy')):
            print('using fixed split')
            train_idx, valid_idx = np.load(f'data_splits/{self.dataset_name}_train_idx.npy'), np.load(f'data_splits/{self.dataset_name}_valid_idx.npy')
            print(len(train_idx),len(valid_idx))
        else:
            print('creating a split')
            indices = list(range(self.num_train))
            train_idx, valid_idx = train_test_split(indices, test_size=self.valid_size)
            np.save(f'data_splits/{self.dataset_name}_train_idx.npy',train_idx)
            np.save(f'data_splits/{self.dataset_name}_valid_idx.npy',valid_idx)
        return train_idx, valid_idx


class TinyImageNet(Dataset):
    def __init__(self, root, train, transform, download=True):

        self.url = "http://cs231n.stanford.edu/tiny-imagenet-200"
        self.root = root
        if download:
            if os.path.exists(f'{self.root}/tiny-imagenet-200/'):
                print('File already downloaded')
            else:
                download_and_extract_archive(self.url, root, filename="tiny-imagenet-200.zip")

        self.root = os.path.join(self.root, "tiny-imagenet-200")
        self.train = train
        self.transform = transform
        self.ids_string = np.sort(np.loadtxt(f"{self.root}/wnids.txt", "str"))
        self.ids = {class_string: i for i, class_string in enumerate(self.ids_string)}
        if train:
            self.paths = glob.glob(f"{self.root}/train/*/images/*")
            self.label = [self.ids[path.split("/")[-3]] for path in self.paths]
        else:
            self.val_annotations = np.loadtxt(f"{self.root}/val/val_annotations.txt", "str")
            self.paths = [f"{self.root}/val/images/{sample[0]}" for sample in self.val_annotations]
            self.label = [self.ids[sample[1]] for sample in self.val_annotations]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = pil_loader(self.paths[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, self.label[idx]