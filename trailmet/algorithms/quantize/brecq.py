
import torch
import torch.nn as nn
from trailmet.utils import seed_everything
from trailmet.algorithms.quantize.quantize import BaseQuantization
from trailmet.algorithms.quantize.quant_model import QuantModel, BaseQuantBlock, QuantModule
from trailmet.algorithms.quantize.reconstruct import layer_reconstruction, block_reconstruction


seed_everything(42)

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
        self.set_8bit_head_stem = self.kwargs.get('SET_8BIT_HEAD_STEM', True)
        self.num_samples = self.kwargs.get('NUM_SAMPLES', 1024)
        self.weight = self.kwargs.get('WEIGHT', 0.01)
        self.iters_w = self.kwargs.get('ITERS_W', 20000)
        self.iters_a = self.kwargs.get('ITERS_A', 20000)
        self.optimizer = self.kwargs.get('OPTIMIZER', 'adam')
        self.lr = self.kwargs.get('LR', 4e-4)
        self.gpu_id = self.kwargs.get('GPU_ID', 0)
        self.calib_bs = self.kwargs.get('CALIB_BS', 64)
        self.p = 2.4         # Lp norm minimization for LSQ
        self.b_start = 20    # temperature at the beginning of calibration
        self.b_end = 2       # temperature at the end of calibration
        self.test_before_calibration = True
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        torch.cuda.set_device(self.gpu_id)


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


    