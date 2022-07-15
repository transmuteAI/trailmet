
from ..algorithms import BaseAlgorithm

class BaseQuantization(BaseAlgorithm):
    """base class for quantization algorithms"""
    def __init__(self, **kwargs):
        super(BaseQuantization, self).__init__(**kwargs)
        pass

    def quantize(self, model, dataloaders, **kwargs):
        pass

    
