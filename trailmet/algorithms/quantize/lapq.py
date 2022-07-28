
import torch
import torch.nn as nn
from trailmet.utils import seed_everything
from trailmet.algorithms.quantize.quantize import BaseQuantization
from trailmet.algorithms.quantize.quant_model import QuantModel, QuantModule
import scipy.optimize as optim


class LAPQ(BaseQuantization):
    def __init__(self, model: nn.Module, dataloaders, **kwargs):
        super(LAPQ, self).__init__(**kwargs)
        self.model = model
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.kwargs = kwargs
        self.w_bits = 8
        self.a_bits = 8
        self.num_samples = 1024
        self.act_quant = True
        self.channel_wise = False   # currently only layer-wise supported
        self.set_8bit_head_stem = True
        self.test_before_calibration = True
        self.maxiter = None
        self.gpu_id = 0
        self.seed = 42
        seed_everything(self.seed)
        self.device = torch.device('cuda.{}'.format(self.gpu_id))
        self.calib_data = self.get_calib_samples(self.train_loader, self.num_samples)

    def compress_model(self):
        wq_params = {'n_bits': self.w_bits, 'channel_wise': self.channel_wise, 'scale_method': 'mse'}
        aq_params = {'n_bits': self.a_bits, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': self.act_quant}
        self.model = self.model.to(self.device)
        self.model.eval()
        self.qnn = QuantModel(model=self.model, weight_quant_params=wq_params, act_quant_params=aq_params)
        self.qnn = self.qnn.to(self.device)

        if self.set_8bit_head_stem:
            print('==> Setting the first and the last layer to 8-bit')
            self.qnn.set_first_last_layer_to_8bit()
        
        self.qnn.set_quant_state(True, False)
        print('==> Initializing weight quantization parameters')
        _ = self.qnn(self.calib_data[:16].to(self.device))
        if self.test_before_calibration:
            print('Quantized accuracy before lapq: {}'.format(self.test(self.qnn, self.val_loader, device=self.device)))
        
        # To do : add support for scale initialization based on quadratic interpolation 
        ######### of Lp norm instead of minimizing mse

        init_scales = []
        for module in self.qnn.modules():
            if isinstance(module, QuantModule):
                init_scales.append(torch.squeeze(module.weight_quantizer.get_scales()[0]).item()) 
                init_scales.append(torch.squeeze(module.weight_quantizer.get_scales()[1]).item())

        min_method = "Powell"
        min_options = {}
        if self.maxiter is not None:
            min_options['maxiter'] = self.maxiter
        res = optim.minimize(lambda scales: self.evaluate_calibration_clipped(scales, self.qnn),
                             init_scales, method=min_method, options=min_options)
        scales = res.x
        self.qnn.set_quant_params(scales)
        self.qnn.set_quant_state(weight_quant=True, act_quant=True)
        print('Full quantization (W{}A{}) accuracy: {}'.format(self.w_bits, self.a_bits, 
                            self.test(self.qnn, self.val_loader, device=self.device))) 

    def eval_pnorm_on_calibration(p):
        pass

    def evaluate_calibration_clipped(self, scales, q_model: QuantModel):
        q_model.set_quant_params(scales)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        _, _, loss = self.test(q_model, self.calib_data, criterion, device=self.device)
        return loss

        
