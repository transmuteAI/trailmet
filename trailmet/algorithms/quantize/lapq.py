
import copy
import torch
import torch.nn as nn
import numpy as np
import scipy.optimize as optim
from itertools import count
from tqdm import tqdm as tqdm_notebook
from trailmet.utils import seed_everything
from trailmet.algorithms.quantize.quantize import BaseQuantization
from trailmet.algorithms.quantize.qmodel import ModelQuantizer


class LAPQ(BaseQuantization):
    def __init__(self, model: nn.Module, dataloaders, **kwargs):
        super(LAPQ, self).__init__(**kwargs)
        self.model = model
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.kwargs = kwargs
        self.w_bits = kwargs.get('W_BITS', 8)
        self.a_bits = kwargs.get('A_BITS', 8)
        self.calib_batches = kwargs.get('CALIB_BATCHES', 16)
        self.act_quant = kwargs.get('ACT_QUANT', True)
        self.test_before_calibration = True
        self.maxiter = kwargs.get('MAX_ITER', 1)
        self.maxfev = kwargs.get('MAX_FEV', 1)
        self.verbose = kwargs.get('VERBOSE', True)
        self.print_freq = kwargs.get('PRINT_FREQ', 20)
        self.gpu_id = kwargs.get('GPU_ID', 0)
        self.seed = kwargs.get('SEED', 42)
        seed_everything(self.seed)
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        self.calib_data = self.get_calib_samples(self.train_loader, 64*self.calib_batches)
        self.eval_count = count(0)
        self.min_loss = 1e6

    def compress_model(self):
        self.model.to(self.device)
        self.search_absorbe_bn(self.model)
        args = {
            'bit_weights' : self.w_bits,
            'bit_act' : self.a_bits,
            'bcorr_w' : True,
            'qtype' : 'lp_norm',
            'lp' : 2.0
        }
        layers = []
        layers += [n for n, m in self.model.named_modules() if isinstance(m, nn.Conv2d)][1:-1]
        if self.act_quant:
            layers += [n for n, m in self.model.named_modules() if isinstance(m, nn.ReLU)][1:-1]
            layers += [n for n, m in self.model.named_modules() if isinstance(m, nn.ReLU6)][1:-1]

        args['qtype'] = 'max_static'
        cnn = copy.deepcopy(self.model)
        mq = ModelQuantizer(cnn, args, layers)
        print('Quantization (W{}A{}) accuracy before LAPQ: {}'.format(
            self.w_bits, self.a_bits, 
            self.test(mq.model, self.val_loader, device=self.device)))
        del mq, cnn

        ps = np.linspace(2,4,10)
        losses = []
        tk1=tqdm_notebook(ps, total=len(ps))
        for p in tk1:
            args['qtype'] = 'lp_norm'
            args['lp'] = p
            cnn = copy.deepcopy(self.model)
            mq = ModelQuantizer(cnn, args, layers)
            loss = self.evaluate_loss(model=mq.model, device=self.device)
            losses.append(loss.item())
            tk1.set_postfix(p_val=p, loss=loss.item())
            del mq, cnn
        # using quadratic interpolation to approximate the optimal quantization step size ∆p∗
        z = np.polyfit(ps, losses, 2)
        y = np.poly1d(z)
        p_intr = y.deriv().roots[0]

        args['lp'] = p_intr
        quant_model = ModelQuantizer(self.model, args, layers)
        lp_acc1, lp_acc5, lp_loss = self.test(quant_model.model, self.val_loader, torch.nn.CrossEntropyLoss().to(self.device), self.device)
        lp_point = quant_model.get_clipping()

        if self.verbose:
            print("==> p intr : {:.2f}".format(p_intr))
            print("==> loss : {:.4f}".format(lp_loss))
            print("==> acc@1 | acc@5 : {:.4f} | {:.4f}".format(lp_acc1, lp_acc5))

        min_method = "Powell"
        min_options = {
            'maxiter' : self.maxiter,
            'maxfev' : self.maxfev
        }
        init_scale = lp_point.cpu().numpy()
        count_iter = count(0)
        def local_search_callback(x):
            it = next(count_iter)
            quant_model.set_clipping(x, self.device)
            loss = self.evaluate_loss(quant_model, self.device)
            print('\n==> Loss at end of iter [{}] : {:.4f}\n'.format(it, loss))
            if self.verbose:
                print('==> Layer-wise Scales :\n', x)

        res = optim.minimize(
            lambda scales: self.evaluate_calibration(scales, quant_model, self.device), init_scale,
            method=min_method, options=min_options, callback=local_search_callback
        )
        scales = res.x
        quant_model.set_clipping(scales, self.device)
        print('Full quantization (W{}A{}) accuracy: {}'.format(
            self.w_bits, self.a_bits, 
            self.test(quant_model, self.val_loader, device=self.device)))
        self.qnn = copy.deepcopy(quant_model.model)


    def evaluate_calibration(self, scales, QM: ModelQuantizer, device):
        eval_count = next(self.eval_count)
        QM.set_clipping(scales, device)
        loss = self.evaluate_loss(QM.model, device).item()
        if loss < self.min_loss:
            self.min_loss = loss
        if self.verbose and eval_count%self.print_freq==0:
            print("==> iteration: {}, minimum loss so far: {:.4f}".format(
            eval_count, self.min_loss))
        return loss

    def evaluate_loss(self, model: nn.Module, device):
        criterion = torch.nn.CrossEntropyLoss().to(device)
        model.eval()
        with torch.no_grad():
            if not hasattr(self, 'cal_set'):
                self.cal_set = []
                for i, (images, target) in enumerate(self.train_loader):
                    if i>=self.calib_batches:             # TODO: make this robust for variable batch size
                        break
                    images = images.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    self.cal_set.append((images, target))
            res = torch.tensor([0.]).to(device)
            for i in range(len(self.cal_set)):
                images, target = self.cal_set[i]
                output = model(images)
                loss = criterion(output, target)
                res += loss
            return res / len(self.cal_set)        
