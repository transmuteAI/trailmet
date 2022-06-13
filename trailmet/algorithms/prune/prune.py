

from ..algorithms import BaseAlgorithm


class BasePruning(BaseAlgorithm):

    def __init__(self):
        super(BasePruning, self).__init__()

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
