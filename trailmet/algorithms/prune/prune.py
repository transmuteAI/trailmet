from ..algorithms import BaseAlgorithm

class BasePruning(BaseAlgorithm):
    """base class for pruning algorithms"""
    def __init__(self, **kwargs):
        super(BasePruning, self).__init__(**kwargs)
        pass
    
    def prune(self, model, dataloaders, **kwargs):
        pass
