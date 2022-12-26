import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

class BaseDataset:
    """
    Base class to be inherited by 
    """
    def __init__(self, name=None, root=None,
                transform=None,
                target_transform=None,
                 download=True, 
                 split_types = None,
                 val_fraction = 0.2,
                 shuffle=True,
                 random_seed = 42):

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
            self.val_fraction = float(self.val_fraction) # check for float type
        except ValueError:
            raise ValueError('val_fraction needs to be of float type')

        assert (self.val_fraction >= 0.0) and (self.val_fraction < 1.0), 'val_fraction should be in range [0.0, 1.0]'

        # check if split_types contain nothing beyond train, val and test
        assert (len(set(self.split_types) - set(['train', 'val', 'test'])) == 0), 'Only train, val and test permitted as inputs in split_types'

        if self.shuffle:
            # ensure that val exists in split_types
            assert 'val' in self.split_types, 'Required param val missing in the defined split types (split_type)'

    def stack_dataset(self):
        """
        Returns
        -------
            dataset_dict (dictionary): The keys of the dictionary are "train_datset", "val_dataset"
            and "test_dataset" and the values are object of pytorch CIFER10 dataset containing train,
            val and test respectively.
        """
        num_train = len(self.dataset_dict["train_dataset"])
        indices = list(range(num_train))
        split = int(np.floor(self.val_fraction * num_train))

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        self.dataset_dict["train_sampler"] = train_sampler
        self.dataset_dict["val_sampler"] = valid_sampler
        self.dataset_dict["test_sampler"] = None
        return self.dataset_dict
