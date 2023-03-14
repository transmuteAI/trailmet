
import torch
import torch.nn as nn
import torch.distributed as dist
from trailmet.utils import seed_everything
from trailmet.algorithms.quantize.quantize import BaseQuantization, FoldBN, StraightThrough
from trailmet.models.resnet import BasicBlock, Bottleneck
from trailmet.models.mobilenet import InvertedResidual
from trailmet.algorithms.quantize.qmodel import QuantModule, BaseQuantBlock
from trailmet.algorithms.quantize.qmodel import QuantBasicBlock, QuantBottleneck, QuantInvertedResidual
from trailmet.algorithms.quantize.reconstruct import layer_reconstruction, block_reconstruction

supported = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    InvertedResidual: QuantInvertedResidual,
}

class BRECQ(BaseQuantization):
    """
    Class for post-training quantization of models using block reconstruction 
    method based on - BRECQ: PUSHING THE LIMIT OF POST-TRAINING QUANTIZATION 
    BY BLOCK RECONSTRUCTION :- https://arxiv.org/abs/2102.05426 
    :param W_BITS: bitwidth for weight quantization
    :param A_BITS: bitwidth for activation quantization
    :param CHANNEL_WISE: apply channel_wise quantization for weights
    :param ACT_QUANT: apply activation quantization
    :param SET_8BIT_HEAD_STEM: Set the first and the last layer to 8-bit
    :param NUM_SAMPLES: size of calibration dataset
    :param WEIGHT: weight of rounding cost vs the reconstruction loss
    :param ITERS_W: number of iteration for AdaRound
    :param ITERS_A: number of iteration for LSQ
    :params LR: learning rate for LSQ
    """
    def __init__(self, model: nn.Module, dataloaders, **kwargs):
        super(BRECQ, self).__init__(**kwargs)
        self.model = model
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.kwargs = kwargs
        self.w_bits = self.kwargs.get('W_BITS', 8)
        self.a_bits = self.kwargs.get('A_BITS', 8)
        self.channel_wise = self.kwargs.get('CHANNEL_WISE', True)
        self.act_quant = self.kwargs.get('ACT_QUANT', True)
        self.set_8bit_head_stem = self.kwargs.get('SET_8BIT_HEAD_STEM', False)
        self.precision_config = self.kwargs.get('PREC_CONFIG', [])
        self.num_samples = self.kwargs.get('NUM_SAMPLES', 1024)
        self.weight = self.kwargs.get('WEIGHT', 0.01)
        self.iters_w = self.kwargs.get('ITERS_W', 10000)
        self.iters_a = self.kwargs.get('ITERS_A', 10000)
        self.optimizer = self.kwargs.get('OPTIMIZER', 'adam')
        self.lr = self.kwargs.get('LR', 4e-4)
        self.gpu_id = self.kwargs.get('GPU_ID', 0)
        self.calib_bs = self.kwargs.get('CALIB_BS', 64)
        self.seed = self.kwargs.get('SEED', 42)
        self.p = 2.4         # Lp norm minimization for LSQ
        self.b_start = 20    # temperature at the beginning of calibration
        self.b_end = 2       # temperature at the end of calibration
        self.test_before_calibration = True
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        torch.cuda.set_device(self.gpu_id)
        seed_everything(self.seed)
        print('==> Using seed :',self.seed)


    def compress_model(self):
        """
        method to build quantization parameters and finetune weights and/or activations
        """
        wq_params = {'n_bits': self.w_bits, 'channel_wise': self.channel_wise, 'scale_method': 'mse'}
        aq_params = {'n_bits': self.a_bits, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': self.act_quant}
        self.model = self.model.to(self.device)
        self.model.eval()
        self.qnn = QuantModel(model=self.model, weight_quant_params=wq_params, act_quant_params=aq_params)
        self.qnn = self.qnn.to(self.device)
        self.qnn.eval()

        for i in range(len(self.precision_config)):
            conf = self.precision_config[i]
            self.qnn.set_layer_precision(conf[2], conf[3], conf[0], conf[1])
            print(f'==> Layers from {conf[0]} to {conf[1]} set to precision w{conf[2]}a{conf[3]}')
        
        if self.set_8bit_head_stem:
            print('==> Setting the first and the last layer to 8-bit')
            self.qnn.set_first_last_layer_to_8bit()
        
        self.cali_data = self.get_calib_samples(self.train_loader, self.num_samples)
        # device = next(self.qnn.parameters()).device
        
        # Initialiaze weight quantization parameters 
        self.qnn.set_quant_state(True, False)
        print('==> Initializing weight quantization parameters')
        _ = self.qnn(self.cali_data[:self.calib_bs].to(self.device))
        if self.test_before_calibration:
            print('Quantized accuracy before brecq: {}'.format(self.test(self.qnn, self.val_loader, device=self.device)))
        
        # Start weight calibration
        kwargs = dict(cali_data=self.cali_data, iters=self.iters_w, weight=self.weight, asym=True,
                  b_range=(self.b_start, self.b_end), warmup=0.2, act_quant=False, opt_mode='mse', optim=self.optimizer)
        print('==> Starting weight calibration')
        self.reconstruct_model(self.qnn, **kwargs)
        self.qnn.set_quant_state(weight_quant=True, act_quant=False)
        print('Weight quantization accuracy: {}'.format(self.test(self.qnn, self.val_loader, device=self.device)))

        if self.act_quant:
            # Initialize activation quantization parameters
            self.qnn.set_quant_state(True, True)
            with torch.no_grad():
                _ = self.qnn(self.cali_data[:self.calib_bs].to(self.device))
            
            # Disable output quantization because network output
            # does not get involved in further computation
            self.qnn.disable_network_output_quantization()
            
            # Start activation rounding calibration
            kwargs = dict(cali_data=self.cali_data, iters=self.iters_a, act_quant=True, opt_mode='mse', lr=self.lr, p=self.p, optim=self.optimizer)
            self.reconstruct_model(self.qnn, **kwargs)
            self.qnn.set_quant_state(weight_quant=True, act_quant=True)
            print('Full quantization (W{}A{}) accuracy: {}'.format(self.w_bits, self.a_bits, self.test(self.qnn, self.val_loader, device=self.device))) 


    def reconstruct_model(self, model: nn.Module, **kwargs):
        """
        Method for model parameters reconstruction. Takes in quantized model
        and optimizes weights by applying layer-wise reconstruction for first 
        and last layer, and block reconstruction otherwise.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(self.qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    block_reconstruction(self.qnn, module, **kwargs)
            else:
                self.reconstruct_model(module, **kwargs)


class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.model = model
        bn = FoldBN()
        bn.search_fold_and_remove_bn(self.model)
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if type(child_module) in supported:
                setattr(module, name, supported[type(child_module)](child_module, weight_quant_params, act_quant_params))

            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, input):
        return self.model(input)

    def set_first_last_layer_to_8bit(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[0].weight_quantizer.bitwidth_refactor(8)
        module_list[0].act_quantizer.bitwidth_refactor(8)
        module_list[-1].weight_quantizer.bitwidth_refactor(8)
        module_list[-2].act_quantizer.bitwidth_refactor(8)
        # ignore reconstruction of the first layer
        module_list[0].ignore_reconstruction = True

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True

    def set_layer_precision(self, weight_bit=8, act_bit=8, start=1, end=-1):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        assert start>=0 and end>=0, 'layer index cannot be negative'
        assert start<len(module_list) and end<len(module_list), 'layer index out of range'
        for i in range(start, end+1):
            module_list[i].weight_quantizer.bitwidth_refactor(weight_bit)
            if i==len(module_list)-1: continue
            module_list[i].act_quantizer.bitwidth_refactor(act_bit)

    def synchorize_activation_statistics(self):
        for m in self.modules():
            if isinstance(m, QuantModule):
                if m.act_quantizer.delta is not None:
                    m.act_quantizer.delta.data /= dist.get_world_size()
                    dist.all_reduce(m.act_quantizer.delta.data) 