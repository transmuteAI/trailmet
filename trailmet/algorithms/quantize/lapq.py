
import copy
import torch
import torch.nn as nn
import numpy as np
import scipy.optimize as optim
from tqdm import tqdm
from itertools import count
from trailmet.utils import seed_everything
from trailmet.algorithms.quantize.quantize import BaseQuantization, Conv2dFunctor, LinearFunctor
from trailmet.algorithms.quantize.methods import LearnedStepSizeQuantization, FixedClipValueQuantization
from trailmet.algorithms.quantize.qmodel import ParameterModuleWrapper, ActivationModuleWrapper


class LAPQ(BaseQuantization):
    def __init__(self, model: nn.Module, dataloaders, **kwargs):
        super(LAPQ, self).__init__(**kwargs)
        self.model = model
        self.train_loader = dataloaders['train']
        self.test_loader = dataloaders['test']
        self.kwargs = kwargs
        self.w_bits = kwargs.get('W_BITS', 8)
        self.a_bits = kwargs.get('A_BITS', 8)
        self.calib_batches = kwargs.get('CALIB_BATCHES', 16)
        self.act_quant = kwargs.get('ACT_QUANT', True)
        self.test_before_calibration = kwargs.get('DRY_RUN', True)
        self.maxiter = kwargs.get('MAX_ITER', 1)
        self.maxfev = kwargs.get('MAX_FEV', 1)
        self.verbose = kwargs.get('VERBOSE', True)
        self.print_freq = kwargs.get('PRINT_FREQ', 20)
        self.gpu_id = kwargs.get('GPU_ID', 0)
        self.seed = kwargs.get('SEED', 42)
        seed_everything(self.seed)
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        if self.verbose:
            print("==> Using seed: {} and device: cuda:{}".format(self.seed, self.gpu_id))
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

        if self.test_before_calibration:
            args['qtype'] = 'max_static'
            cnn = copy.deepcopy(self.model)
            qm = QuantModel(cnn, args, layers)
            acc1, acc5 = self.test(qm.model, self.test_loader, device=self.device)
            print('==> Quantization (W{}A{}) accuracy before LAPQ: {:.4f} | {:.4f}'.format(
                self.w_bits, self.a_bits, acc1, acc5))
            del qm, cnn

        ps = np.linspace(2,4,10)
        losses = []
        tk1=tqdm(ps, total=len(ps))
        for p in tk1:
            args['qtype'] = 'lp_norm'
            args['lp'] = p
            cnn = copy.deepcopy(self.model)
            qm = QuantModel(cnn, args, layers)
            loss = self.evaluate_loss(model=qm.model, device=self.device)
            losses.append(loss.item())
            tk1.set_postfix(p_val=p, loss=loss.item())
            del qm, cnn
        # using quadratic interpolation to approximate the optimal quantization step size ∆p∗
        z = np.polyfit(ps, losses, 2)
        y = np.poly1d(z)
        p_intr = y.deriv().roots[0]

        print("==> using p intr : {:.2f}".format(p_intr))
        args['lp'] = p_intr
        quant_model = QuantModel(self.model, args, layers)
        lp_acc1, lp_acc5, lp_loss = self.test(quant_model.model, self.test_loader, torch.nn.CrossEntropyLoss().to(self.device), self.device)
        lp_point = quant_model.get_clipping()

        if self.verbose:
            print('==> Quantization (W{}A{}) accuracy before Optimization: {:.4f} | {:.4f}'.format(
                self.w_bits, self.a_bits, lp_acc1, lp_acc5))
            print("==> Loss after LpNormQuantization: {:.4f}".format(lp_loss))
            print("==> Starting Powell Optimization")

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
            loss = self.evaluate_loss(quant_model.model, self.device)
            if self.verbose:
                print('\n==> Loss at end of iter [{}] : {:.4f}\n'.format(it, loss.item()))

        self.pbar = tqdm(total=min(self.maxiter, self.maxfev))
        res = optim.minimize(
            lambda scales: self.evaluate_calibration(scales, quant_model, self.device), init_scale,
            method=min_method, options=min_options, callback=local_search_callback
        )
        self.pbar.close()
        scales = res.x
        if self.verbose:
            print('==> Layer-wise Scales :\n', scales)
        quant_model.set_clipping(scales, self.device)
        print('==> Full quantization (W{}A{}) accuracy: {}'.format(
            self.w_bits, self.a_bits, 
            self.test(quant_model.model, self.test_loader, device=self.device)))
        self.qnn = copy.deepcopy(quant_model.model)
        return self.qnn


    def evaluate_calibration(self, scales, QM, device):
        eval_count = next(self.eval_count)
        QM.set_clipping(scales, device)
        loss = self.evaluate_loss(QM.model, device).item()
        if loss < self.min_loss:
            self.min_loss = loss
        # if self.verbose and eval_count%self.print_freq==0:
        #     print("==> iteration: {}, minimum loss so far: {:.4f}".format(
        #     eval_count, self.min_loss))
        self.pbar.set_postfix(curr_loss=loss, min_loss=self.min_loss)
        self.pbar.update(1)
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


class QuantModel:
    def __init__(self, model, args, quantizable_layers, optimizer_bridge=None):
        self.model = model
        self.args = args
        self.bit_weights = args['bit_weights']
        self.bit_act = args['bit_act']
        self.post_relu = True
        
        self.replacement_factory = {
            nn.ReLU: ActivationModuleWrapper,
            nn.ReLU6: ActivationModuleWrapper,
            nn.Conv2d: ParameterModuleWrapper 
            }
        self.functor_map = {
            nn.Conv2d: Conv2dFunctor, 
            nn.Linear: LinearFunctor, 
            }
        self.optimizer_bridge = optimizer_bridge
        
        self.quantization_wrappers = []
        self.quantizable_modules = []
        self.quantizable_layers = quantizable_layers
        self._pre_process_container(model)
        self._create_quantization_wrappers()
        self.quantization_params = LearnedStepSizeQuantization.learned_parameters()

    def load_state_dict(self, state_dict):
        for name, qwrapper in self.quantization_wrappers:
            qwrapper.load_state_dict(state_dict)

    def freeze(self):
        for n, p in self.model.named_parameters():
            # TODO: hack, make it more robust
            if not np.any([qp in n for qp in self.quantization_params]):
                p.requires_grad = False

    @staticmethod
    def has_children(module):
        try:
            next(module.children())
            return True
        except StopIteration:
            return False
    
    def _create_quantization_wrappers(self):
        for qm in self.quantizable_modules:
            # replace module by it's wrapper
            fn = self.functor_map[type(qm.module)](qm.module) if type(qm.module) in self.functor_map else None
            args = {"bits_out": self.bit_act, "bits_weight": self.bit_weights, "forward_functor": fn,
                    "post_relu": self.post_relu, "optim_bridge": self.optimizer_bridge}
            args.update(self.args)
            if hasattr(qm, 'bn'):
                args['bn'] = qm.bn
            module_wrapper = self.replacement_factory[type(qm.module)](qm.full_name, qm.module,
                                                                    **args)
            setattr(qm.container, qm.name, module_wrapper)
            self.quantization_wrappers.append((qm.full_name, module_wrapper))

    def _pre_process_container(self, container, prefix=''):
        prev, prev_name = None, None
        for name, module in container.named_children():
            # if is_bn(module) and is_absorbing(prev) and prev_name in self.quantizable_layers:
            #     # Pass BN module to prev module quantization wrapper for BN folding/unfolding
            #     self.quantizable_modules[-1].bn = module

            full_name = prefix + name
            if full_name in self.quantizable_layers:
                self.quantizable_modules.append(
                    type('', (object,), {'name': name, 'full_name': full_name, 'module': module, 'container': container})()
                )

            if self.has_children(module):
                # For container we call recursively
                self._pre_process_container(module, full_name + '.')

            prev = module
            prev_name = full_name

    def get_qwrappers(self):
        return [qwrapper for (name, qwrapper) in self.quantization_wrappers if qwrapper.__enabled__()]

    def set_clipping(self, clipping, device):  # TODO: handle device internally somehow
        qwrappers = self.get_qwrappers()
        for i, qwrapper in enumerate(qwrappers):
            qwrapper.set_quantization(FixedClipValueQuantization,
                                      {'clip_value': clipping[i], 'device': device})

    def get_clipping(self):
        clipping = []
        qwrappers = self.get_qwrappers()
        for i, qwrapper in enumerate(qwrappers):
            q = qwrapper.get_quantization()
            clip_value = getattr(q, 'alpha')
            clipping.append(clip_value.item())

        return qwrappers[0].get_quantization().alpha.new_tensor(clipping)