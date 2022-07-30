
from itertools import count
import torch
import torch.nn as nn
import scipy.optimize as optim
import numpy as np
from tqdm import tqdm
from trailmet.utils import seed_everything
from trailmet.algorithms.quantize.quantize import BaseQuantization
from trailmet.algorithms.quantize.quant_model import QuantModel, QuantModule


class LAPQ(BaseQuantization):
    def __init__(self, model: nn.Module, dataloaders, **kwargs):
        super(LAPQ, self).__init__(**kwargs)
        self.model = model
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.kwargs = kwargs
        self.w_bits = kwargs.get('W_BITS', 8)
        self.a_bits = kwargs.get('A_BITS', 8)
        self.num_samples = kwargs.get('NUM_SAMPLES', 1024)
        self.act_quant = kwargs.get('ACT_QUANT', True)
        self.channel_wise = False   # lapq supports only layer-wise
        self.set_8bit_head_stem = kwargs.get('SET_8BIT_HEAD_STEM', True)
        self.test_before_calibration = True
        self.maxiter = kwargs.get('MAX_ITER', None)
        self.verbose = kwargs.get('VERBOSE', True)
        self.print_freq = kwargs.get('PRINT_FREQ', 20)
        self.gpu_id = kwargs.get('GPU_ID', 0)
        self.seed = kwargs.get('SEED', 42)
        seed_everything(self.seed)
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        self.calib_data = self.get_calib_samples(self.train_loader, self.num_samples)
        self.eval_count = count(0)
        self.min_loss = 1e6


    def compress_model(self):
        # quant params for LAPQ
        wq_params = {
            'num_bits': self.w_bits, 
            'symm': True, 
            'uint': True, 
            'stochastic': False, 
            'tails': False,
            'q_type': 'lp_norm',
            'lp': 2.0
            }
        aq_params = {
            'num_bits': self.a_bits, 
            'symm': True, 
            'uint': True, 
            'stochastic': False, 
            'tails': False,
            'q_type': 'lp_norm',
            'lp': 2.0
            }
        self.model.to(self.device)
        self.model.eval()

        # minimize lp norm of the quantization error of weights and activations 
        # in each layer with respect to clipping values, with different values of p
        ps = np.linspace(2,4,10)
        losses = []
        for p in tqdm():
            wq_params['lp'] = p
            aq_params['lp'] = p
            qm = QuantModel(model=self.model, weight_quant_params=wq_params, act_quant_params=aq_params)
            qm.to(self.device)
            loss = self.evaluate_loss(qm, self.device)
            losses.append(loss.item())
            del qm
            if self.verbose:
                print("(p, loss) - ({}, {})".format(p, loss.item()))
        
        # use quadratic interpolation to approximate the optimal quantization step size ∆p∗
        z = np.polyfit(ps, losses, 2)
        y = np.poly1d(z)
        p_intr = y.deriv().roots[0]

        wq_params['lp'] = p_intr
        aq_params['lp'] = p_intr
        self.qnn = QuantModel(model=self.model, weight_quant_params=wq_params, act_quant_params=aq_params)
        self.qnn.to(self.device)
        if self.set_8bit_head_stem:
            self.qnn.set_first_last_layer_to_8bit()
        self.qnn.set_quant_state(weight_quant=True, act_quant=True)

        lp_acc1, lp_acc5, lp_loss = self.test(self.qnn, self.val_loader, torch.nn.CrossEntropyLoss.to(self.device), self.device)
        lp_point = self.qnn.get_quant_params()
        if self.verbose:
            print("==> p intr : {:.2f}".format(p_intr))
            print("==> loss : {:.4f}".format(lp_loss))
            print("==> acc@1 | acc@5 : {:.4f} | {:.4f}".format(lp_acc1, lp_acc5))

        # use the optimal quantization step size ∆p∗ as a starting point for a gradient-free 
        # joint optimization algorithm, such as Powell’s method, to minimize the loss and find ∆∗.
        init_scales = lp_point
        min_method = "Powell"
        min_options = {}
        if self.maxiter is not None:
            min_options['maxiter'] = self.maxiter
        res = optim.minimize(lambda scales: self.evaluate_calibration_clipped(scales, self.qnn, **wq_params),
                             init_scales.cpu().numpy(), method=min_method, options=min_options)
        scales = res.x
        self.qnn.set_quant_params(scales, self.device, **wq_params)
        print('Full quantization (W{}A{}) accuracy: {}'.format(self.w_bits, self.a_bits, 
                            self.test(self.qnn, self.val_loader, device=self.device))) 


    def evaluate_calibration_clipped(self, scales, q_model: QuantModel, **qargs):
        eval_count = next(self.eval_count)
        q_model.set_quant_params(scales, self.device, **qargs)
        loss = self.evaluate_loss(q_model, self.device)
        if loss<self.min_loss:
            self.min_loss=loss
        if self.verbose and eval_count%self.print_freq==0:
            print("==> eval iteration: {}, minimum loss of last {} iterations: {:.4f}".format(
            eval_count, self.print_freq, self.min_loss))
        return loss

    def evaluate_loss(self, q_model: QuantModel, device):
        criterion = torch.nn.CrossEntropyLoss().to(device)
        q_model.eval()
        with torch.no_grad():
            if not hasattr(self, 'cal_set'):
                self.cal_set = []
                for i, images, target in enumerate(self.train_loader):
                    if i>=16:                       # To do: change this for variable batch size
                        break
                    images = images.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    self.cal_set.append((images, target))

            res = torch.tensor([0.]).to(device)
            for i in range(len(self.cal_set)):
                images, target = self.cal_set[i]
                output = self.model(images)
                loss = criterion(output, target)
                res += loss

            return res / len(self.cal_set)
        






        # self.qnn.set_quant_state(True, False)
        # print('==> Initializing weight quantization parameters')
        # _ = self.qnn(self.calib_data[:16].to(self.device))
        # if self.test_before_calibration:
        #     print('Quantized accuracy before lapq: {}'.format(self.test(self.qnn, self.val_loader, device=self.device)))
        
        # To do : add support for scale initialization based on quadratic interpolation 
        ######### of Lp norm instead of minimizing mse

        # init_scales = []
        # for module in self.qnn.modules():
        #     if isinstance(module, QuantModule):
        #         init_scales.append(torch.squeeze(module.weight_quantizer.get_scales()[0]).item()) 
        #         init_scales.append(torch.squeeze(module.weight_quantizer.get_scales()[1]).item())