import os

class Dataset:
    def __init__(self, name=None, root=None,
                 train=True, transform=None,
                 target_transform=None,
                 download=True):

        self.name = name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        pass