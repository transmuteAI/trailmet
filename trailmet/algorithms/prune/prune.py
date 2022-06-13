

from ..algorithms import BaseAlgorithm


class BasePruning(BaseAlgorithm):

    def __init__(self):
        super(BasePruning, self).__init__()

        # Set default values of parameters generic to all pruning methods
        self.epochs = {'pretrain': 50, 'prune': 20, 'finetune': 20}
        self.optimizer = 'Adam'

        pass

    def pretrain(self):
        pass

    def prune(self):
        pass

    def soft_prune(self):
        pass

    def hard_prune(self):
        pass

    def finetune(self):
        pass
