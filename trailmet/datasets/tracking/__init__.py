from .got10kdata import GOT10kDataset

class TrackingDatasetFactory(object):
    """This class forms the generic wrapper for the different dataset classes.

    The module includes utilities to load datasets, including methods to load
    and fetch popular reference datasets.
    """

    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name(string): dataset name 'CIFAR10', 'CIFAR100', 'ImageNet', 'CHEST',
            root(string):  Root directory of dataset where directory
                   cifar-10-batches-py exists or will be saved
                   to if download is set to True.
        Return:
            dataset(tuple): dataset
        """
        assert 'name' in kwargs, 'should provide dataset name'
        name = kwargs['name']
        assert 'root' in kwargs, 'should provide dataset root'
        if 'got10k' == name:
            obj_dfactory = GOT10kDataset(**kwargs)
        else:
            raise Exception(f"unknown dataset{kwargs['name']}")
        dataset = obj_dfactory.stack_dataset()
        dataset = obj_dfactory.build_dict_info()

        return dataset