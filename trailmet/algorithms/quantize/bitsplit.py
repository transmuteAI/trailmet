


import torch
import copy
import torch.nn as nn
from trailmet.utils import seed_everything
from trailmet.algorithms.quantize.qmodel_bitsplit import QuantModel, Quantizer
from trailmet.algorithms.quantize.quantize import BaseQuantization


class BitSplit(BaseQuantization):
    def __init__(self, model: nn.Module, dataloaders, **kwargs):
        super(BitSplit, self).__init__(**kwargs)
        self.model = model
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.kwargs = kwargs
        self.w_bits = self.kwargs.get('W_BITS', 8)
        self.a_bits = self.kwargs.get('A_BITS', 8)
        self.gpu_id = self.kwargs.get('GPU_ID', 0)
        self.seed = self.kwargs.get('SEED', 42)
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        torch.cuda.set_device(self.gpu_id)
        seed_everything(self.seed)

        
    def compress_model(self):
        self.model.to(self.device)
        self.qmodel = copy.deepcopy(self.model)
        QuantModel(self.qmodel)

        self.act_quant_modules = []
        for m in self.qmodel.modules():
            if isinstance(m, Quantizer):
                m.set_bitwidth(self.a_bits)
                self.act_quant_modules.append(m)
        self.act_quant_modules[-1].set_bitwidth(8)


        #### Weight Quantization ####
        conv = self.model.conv1
        q_conv = self.qmodel.conv1
        # conduct


