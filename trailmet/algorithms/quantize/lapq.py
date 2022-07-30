
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
        self.w_bits = 4
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
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        self.calib_data = self.get_calib_samples(self.train_loader, self.num_samples)

    def compress_model(self):
        # quant params for LAPQ
        wq_params = {
            'num_bits': self.w_bits, 
            'symm': True, 
            'uint': True, 
            'stochastic': False, 
            'tails': False,
            'q_type': 'lp_norm'
            }
        aq_params = {
            'num_bits': self.a_bits, 
            'symm': True, 
            'uint': True, 
            'stochastic': False, 
            'tails': False,
            'q_type': 'lp_norm'
            }
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
        loss = self.evaluate_loss(q_model, criterion)
        return loss

    def evaluate_loss(self, q_model: QuantModel, criterion):
        q_model.eval()
        with torch.no_grad():
            if not hasattr(self, 'cal_set'):
                self.cal_set = []
                for i, images, target in enumerate(self.train_loader):
                    if i>=16:
                        break
                    images = images.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    self.cal_set.append((images, target))

            res = torch.tensor([0.]).to(self.device)
            for i in range(len(self.cal_set)):
                images, target = self.cal_set[i]
                output = self.model(images)
                loss = criterion(output, target)
                res += loss

            return res / len(self.cal_set)
        
