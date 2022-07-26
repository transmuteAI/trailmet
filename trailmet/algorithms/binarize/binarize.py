from ..algorithms import BaseAlgorithm


class BaseBinarize(BaseAlgorithm):
    """base class for binarization algorithms"""
    def __init__(self, **kwargs):
        super(BaseBinarize, self).__init__(**kwargs)
        pass
    
    def Binarize(self, model, dataloaders, **kwargs):
        pass