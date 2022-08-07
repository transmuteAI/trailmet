import torch,os,sys
import torch.nn as nn
from trailmet.utils import seed_everything
from trailmet.algorithms.quantize.quantize import BaseQuantization
from trailmet.algorithms.quantize.quantize import FoldBN
import os, sys
proj_root_dir = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(proj_root_dir)
sys.path.append('../../')
import scipy.optimize as opt
from pathlib import Path
import numpy as np
from itertools import count
from tqdm import tqdm
from trailmet.algorithms.quantize.methods import FixedClipValueQuantization, MaxAbsStaticQuantization, AciqLaplaceQuantization, AciqGausQuantization, MseUniformPriorQuantization, MseNoPriorQuantization, AngDistanceQuantization, L3NormQuantization, L2NormQuantization, L1NormQuantization, LpNormQuantization, LogLikeQuantization, KmeansQuantization, LearnedStepSizeQuantization, FixedClipValueQuantization

_eval_count = count(0)
_min_loss = 1e6
quantization_mapping = {'max_static': MaxAbsStaticQuantization,
                        'aciq_laplace': AciqLaplaceQuantization,
                        'aciq_gaus': AciqGausQuantization,
                        'mse_uniform_prior': MseUniformPriorQuantization,
                        'mse_no_prior': MseNoPriorQuantization,
                        'ang_dis': AngDistanceQuantization,
                        'l3_norm': L3NormQuantization,
                        'l2_norm': L2NormQuantization,
                        'l1_norm': L1NormQuantization,
                        'lp_norm': LpNormQuantization,
                        'log_like': LogLikeQuantization
                        }

class ModelQuantizer:
    def __init__(self, model, quantizable_layers, replacement_factory, optimizer_bridge=None, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.bit_weights = kwargs.get("bit_weights")
        self.bit_act = kwargs.get("bit_act")
        self.post_relu = True
        self.functor_map = {nn.Conv2d: Conv2dFunctor, nn.Linear: LinearFunctor, nn.Embedding: EmbeddingFunctor}
        self.replacement_factory = replacement_factory

        self.optimizer_bridge = optimizer_bridge

        self.quantization_wrappers = []
        self.quantizable_modules = []
        self.quantizable_layers = quantizable_layers
        self._pre_process_container(model)
        self._create_quantization_wrappers()

        # TODO: hack, make it generic
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
            kwargs = {"bits_out": self.bit_act, "bits_weight": self.bit_weights, "forward_functor": fn,
                    "post_relu": self.post_relu, "optim_bridge": self.optimizer_bridge}
            kwargs.update(vars(self.kwargs))
            if hasattr(qm, 'bn'):
                kwargs['bn'] = qm.bn
            module_wrapper = self.replacement_factory[type(qm.module)](qm.full_name, qm.module,
                                                                    **kwargs)
            setattr(qm.container, qm.name, module_wrapper)
            self.quantization_wrappers.append((qm.full_name, module_wrapper))

    def _pre_process_container(self, container, prefix=''):
        prev, prev_name = None, None
        for name, module in container.named_children():
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

    def log_quantizer_state(self, ml_logger, step):
        if self.bit_weights is not None or self.bit_act is not None:
            with torch.no_grad():
                for name, qwrapper in self.quantization_wrappers:
                    qwrapper.log_state(step, ml_logger)

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
        print("len: ",len(qwrappers))
        for i, qwrapper in enumerate(qwrappers):
            q = qwrapper.get_quantization()
            clip_value = getattr(q, 'alpha')
            clipping.append(clip_value.item())

        return qwrappers[0].get_quantization().alpha.new_tensor(clipping)

    class QuantMethod:
        def __init__(self, quantization_wrappers, method):
            self.quantization_wrappers = quantization_wrappers
            self.method = method

        def __enter__(self):
            for n, qw in self.quantization_wrappers:
                qw.set_quant_method(self.method)

        def __exit__(self, exc_type, exc_val, exc_tb):
            for n, qw in self.quantization_wrappers:
                qw.set_quant_method()

    class QuantMode:
        def __init__(self, quantization_wrappers, mode):
            self.quantization_wrappers = quantization_wrappers
            self.mode = mode

        def __enter__(self):
            for n, qw in self.quantization_wrappers:
                qw.set_quant_mode(self.mode)

        def __exit__(self, exc_type, exc_val, exc_tb):
            for n, qw in self.quantization_wrappers:
                qw.set_quant_mode()

    class DisableQuantizer:
        def __init__(self, quantization_wrappers):
            self.quantization_wrappers = quantization_wrappers

        def __enter__(self):
            for n, qw in self.quantization_wrappers:
                qw.active = False

        def __exit__(self, exc_type, exc_val, exc_tb):
            for n, qw in self.quantization_wrappers:
                qw.active = True

    def quantization_method(self, method):
        return ModelQuantizer.QuantMethod(self.quantization_wrappers, method)

    def quantization_mode(self, mode):
        return ModelQuantizer.QuantMode(self.quantization_wrappers, mode)

    def disable(self):
        return ModelQuantizer.DisableQuantizer(self.quantization_wrappers)


class ActivationModuleWrapperPost(nn.Module):
    def __init__(self, name, wrapped_module, **kwargs):
        super(ActivationModuleWrapperPost, self).__init__()
        self.name = name
        self.wrapped_module = wrapped_module
        self.bits_out = kwargs['bits_out']
        self.qtype = kwargs['qtype']
        self.post_relu = True
        self.enabled = True
        self.active = True

        if self.bits_out is not None:
            self.out_quantization = self.out_quantization_default = None

            def __init_out_quantization__(tensor):
                self.out_quantization_default = quantization_mapping[self.qtype](self, tensor, self.bits_out,
                                                                                 symmetric=(not is_positive(wrapped_module)),
                                                                                 uint=True, kwargs=kwargs)
                self.out_quantization = self.out_quantization_default
                print("ActivationModuleWrapperPost - {} | {} | {}".format(self.name, str(self.out_quantization), str(tensor.device)))

            self.out_quantization_init_fn = __init_out_quantization__

    def __enabled__(self):
        return self.enabled and self.active and self.bits_out is not None

    def forward(self, *input):
        if self.post_relu:
            out = self.wrapped_module(*input)

            # Quantize output
            if self.__enabled__():
                self.verify_initialized(self.out_quantization, out, self.out_quantization_init_fn)
                out = self.out_quantization(out)
        else:
            # Quantize output
            if self.__enabled__():
                self.verify_initialized(self.out_quantization, *input, self.out_quantization_init_fn)
                out = self.out_quantization(*input)
            else:
                out = self.wrapped_module(*input)

        return out

    def get_quantization(self):
        return self.out_quantization

    def set_quantization(self, qtype, kwargs, verbose=False):
        self.out_quantization = qtype(self, self.bits_out, symmetric=(not is_positive(self.wrapped_module)),
                                      uint=True, kwargs=kwargs)
        if verbose:
            print("ActivationModuleWrapperPost - {} | {} | {}".format(self.name, str(self.out_quantization),
                                                                      str(kwargs['device'])))

    def set_quant_method(self, method=None):
        if self.bits_out is not None:
            if method == 'kmeans':
                self.out_quantization = KmeansQuantization(self.bits_out)
            else:
                self.out_quantization = self.out_quantization_default

    @staticmethod
    def verify_initialized(quantization_handle, tensor, init_fn):
        if quantization_handle is None:
            init_fn(tensor)

    def log_state(self, step, ml_logger):
        if self.__enabled__():
            if self.out_quantization is not None:
                for n, p in self.out_quantization.named_parameters():
                    if p.numel() == 1:
                        ml_logger.log_metric(self.name + '.' + n, p.item(),  step='auto')
                    else:
                        for i, e in enumerate(p):
                            ml_logger.log_metric(self.name + '.' + n + '.' + str(i), e.item(),  step='auto')


class ParameterModuleWrapperPost(nn.Module):
    def __init__(self, name, wrapped_module, **kwargs):
        super(ParameterModuleWrapperPost, self).__init__()
        self.name = name
        self.wrapped_module = wrapped_module
        self.forward_functor = kwargs['forward_functor']
        self.bit_weights = kwargs['bits_weights']
        self.bits_out = kwargs['bits_out']
        self.qtype = kwargs['qtype']
        self.enabled = True
        self.active = True
        self.centroids_hist = {}
        self.log_weights_hist = False
        self.log_weights_mse = False
        self.log_clustering = False
        self.bn = kwargs['bn'] if 'bn' in kwargs else None
        self.dynamic_weight_quantization = True
        self.bcorr_w = kwargs['bcorr_w']

        setattr(self, 'weight', wrapped_module.weight)
        delattr(wrapped_module, 'weight')
        if hasattr(wrapped_module, 'bias'):
            setattr(self, 'bias', wrapped_module.bias)
            delattr(wrapped_module, 'bias')

        if self.bit_weights is not None:
            self.weight_quantization_default = quantization_mapping[self.qtype](self, self.weight, self.bit_weights,
                                                                             symmetric=True, uint=True, kwargs=kwargs)
            self.weight_quantization = self.weight_quantization_default
            if not self.dynamic_weight_quantization:
                self.weight_q = self.weight_quantization(self.weight)
                self.weight_mse = torch.mean((self.weight_q - self.weight)**2).item()
            print("ParameterModuleWrapperPost - {} | {} | {}".format(self.name, str(self.weight_quantization),
                                                                      str(self.weight.device)))

    def __enabled__(self):
        return self.enabled and self.active and self.bit_weights is not None

    def bias_corr(self, x, xq):
        bias_q = xq.view(xq.shape[0], -1).mean(-1)
        bias_orig = x.view(x.shape[0], -1).mean(-1)
        bcorr = bias_q - bias_orig

        return xq - bcorr.view(bcorr.numel(), 1, 1, 1) if len(x.shape) == 4 else xq - bcorr.view(bcorr.numel(), 1)

    def forward(self, *input):
        w = self.weight
        if self.__enabled__():
            # Quantize weights
            if self.dynamic_weight_quantization:
                w = self.weight_quantization(self.weight)

                if self.bcorr_w:
                    w = self.bias_corr(self.weight, w)
            else:
                w = self.weight_q

        out = self.forward_functor(*input, weight=w, bias=(self.bias if hasattr(self, 'bias') else None))

        return out

    def get_quantization(self):
        return self.weight_quantization

    def set_quantization(self, qtype, kwargs, verbose=False):
        self.weight_quantization = qtype(self, self.bit_weights, symmetric=True, uint=True, kwargs=kwargs)
        if verbose:
            print("ParameterModuleWrapperPost - {} | {} | {}".format(self.name, str(self.weight_quantization),
                                                                      str(kwargs['device'])))

    def set_quant_method(self, method=None):
        if self.bit_weights is not None:
            if method is None:
                self.weight_quantization = self.weight_quantization_default
            elif method == 'kmeans':
                self.weight_quantization = KmeansQuantization(self.bit_weights)
            else:
                self.weight_quantization = self.weight_quantization_default

    # TODO: make it more generic
    def set_quant_mode(self, mode=None):
        if self.bit_weights is not None:
            if mode is not None:
                self.soft = self.weight_quantization.soft_quant
                self.hard = self.weight_quantization.hard_quant
            if mode is None:
                self.weight_quantization.soft_quant = self.soft
                self.weight_quantization.hard_quant = self.hard
            elif mode == 'soft':
                self.weight_quantization.soft_quant = True
                self.weight_quantization.hard_quant = False
            elif mode == 'hard':
                self.weight_quantization.soft_quant = False
                self.weight_quantization.hard_quant = True

    def log_state(self, step, ml_logger):
        if self.__enabled__():
            if self.weight_quantization is not None:
                for n, p in self.weight_quantization.loggable_parameters():
                    if p.numel() == 1:
                        ml_logger.log_metric(self.name + '.' + n, p.item(),  step='auto')
                    else:
                        for i, e in enumerate(p):
                            ml_logger.log_metric(self.name + '.' + n + '.' + str(i), e.item(),  step='auto')

            if self.log_weights_hist:
                ml_logger.tf_logger.add_histogram(self.name + '.weight', self.weight.cpu().flatten(),  step='auto')

            if self.log_weights_mse:
                ml_logger.log_metric(self.name + '.mse_q', self.weight_mse,  step='auto')

def is_positive(module):
    return isinstance(module, nn.ReLU) or isinstance(module, nn.ReLU6)

class Conv2dFunctor:
    def __init__(self, conv2d):
        self.conv2d = conv2d

    def __call__(self, *input, weight, bias):
        res = torch.nn.functional.conv2d(*input, weight, bias, self.conv2d.stride, self.conv2d.padding,
                                         self.conv2d.dilation, self.conv2d.groups)
        return res


class LinearFunctor:
    def __init__(self, linear):
        self.linear = linear

    def __call__(self, *input, weight, bias):
        res = torch.nn.functional.linear(*input, weight, bias)
        return res


class EmbeddingFunctor:
    def __init__(self, embedding):
        self.embedding = embedding

    def __call__(self, *input, weight, bias=None):
        res = torch.nn.functional.embedding(
            *input, weight, self.embedding.padding_idx, self.embedding.max_norm,
            self.embedding.norm_type, self.embedding.scale_grad_by_freq, self.embedding.sparse)
        return res


class OptimizerBridge(object):
    def __init__(self, optimizer, settings={'algo': 'SGD', 'dataset': 'imagenet'}):
        self.optimizer = optimizer
        self.settings = settings

    def add_quantization_params(self, all_quant_params):
        key = self.settings['algo'] + '_' + self.settings['dataset']
        if key in all_quant_params:
            quant_params = all_quant_params[key]
            for group in quant_params:
                self.optimizer.add_param_group(group)

def evaluate_calibration_clipped(scales, model, mq):
    global _eval_count, _min_loss
    eval_count = next(_eval_count)

    mq.set_clipping(scales, model.device)
    loss = evaluate_calibration(mq).item()

    if loss < _min_loss:
        _min_loss = loss

    print_freq = 20
    if eval_count % 20 == 0:
        print("func eval iteration: {}, minimum loss of last {} iterations: {:.4f}".format(
            eval_count, print_freq, _min_loss))

    return loss



class LAPQ(BaseQuantization):
    def __init__(self, model: nn.Module, dataloaders, **kwargs):
        super(LAPQ, self).__init__(**kwargs)
        self.model = model
        self.cal_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.kwargs = kwargs
        self.cal_batch_size = kwargs.get('cal_batch_size',None)
        self.cal_set_size = kwargs.get('cal_set_size' ,None)
        self.print_freq = kwargs.get('print-freq',10)
        self.resume = kwargs.get('resume','')
        self.evaluate = kwargs.get('evaluate',True)
        self.pretrained = kwargs.get('pretrained',True)
        self.custom_resnet = kwargs.get('custom_resnet',True)
        self.custom_inception = kwargs.get('custom_inception', True)
        self.seed = kwargs.get('seed',0)
        self.gpu_ids = kwargs.get('gpu_ids',[7])
        self.shuffle = kwargs.get('shuffle', True)
        self.experiment = kwargs.get('experiment','default')
        self.bit_weights = kwargs.get('bit_weights',None)
        self.bit_act = kwargs.get('bit_act',None)
        self.pre_relu = kwargs.get('pre_relu',True)
        self.qtype = kwargs.get('qtype','max_static')
        self.lp = kwargs.get('lp',3.0)
        self.min_method = kwargs.get('min_method','Powell')
        self.maxiter = kwargs.get('maxiter',None)
        self.maxfew = kwargs.get('maxfev',None)
        self.init_method = kwargs.get('init_method','static')
        self.siv = kwargs.get('siv',1. )
        self.bcorr_w = kwargs.get('bcorr_w',True)
        self.device = f'cuda:{self.gpu_ids[0]}'
        self.bn_folding = kwargs.get('bn_folding',True)

    def compress_model(self):
        seed_everything(42)   
        torch.cuda.set_device(self.device)
        home = str(Path.home())
        self.model.to(self.device)
        foldbn = FoldBN()
#         self.model.eval()
        if self.bn_folding:
            print("Applying batch-norm folding ahead of post-training quantization")
            foldbn.search_fold_and_remove_bn(model = self.model)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        layers = []
        first = 1
        last = -1
#         if args.bit_weights is not None:
        layers += [n for n, m in inf_model.model.named_modules() if isinstance(m, nn.Conv2d)][first:last]
#         if args.bit_act is not None:
        layers += [n for n, m in inf_model.model.named_modules() if isinstance(m, nn.ReLU)][first:last]

        replacement_factory = {nn.ReLU: ActivationModuleWrapperPost,
                           nn.ReLU6: ActivationModuleWrapperPost,
                           nn.Conv2d: ParameterModuleWrapperPost}

        def evaluate_calibration(model):
        # switch to evaluate mode
#             model.eval()
    
            with torch.no_grad():
                if not hasattr(self, 'cal_set'):
                    cal_set = []
                    # TODO: Workaround, refactor this later
                    for i, (images, target) in enumerate(self.cal_loader):
                        if i * self.cal_batch_size >= self.cal_set_size:
                            break
                        images = images.to(self.device, non_blocking=True)
                        target = target.to(self.device, non_blocking=True)
                        cal_set.append((images, target))

                res = torch.tensor([0.]).to(self.device)
                for i in range(len(cal_set)):
                    images, target = cal_set[i]
                    # compute output
                    output = self.model(images)
                    loss = self.criterion(output, target)
                    res += loss

                return res / len(cal_set)

        mq = ModelQuantizer(self.model, layers, replacement_factory, **self.kwargs)
        maxabs_loss = evaluate_calibration(mq)
        print("max loss: {:.4f}".format(maxabs_loss.item()))
        max_point = mq.get_clipping()

        # evaluate
        maxabs_acc = 0
        data = {'max': {'alpha': max_point.cpu().numpy(), 'loss': maxabs_loss.item(), 'acc': maxabs_acc}}

        del mq
        
        def eval_pnorm(p):
            self.qtype = 'lp_norm'
            self.lp = p
            mq = ModelQuantizer(self.model, layers, replacement_factory, **self.kwargs)
            loss = evaluate_calibration(mp)
            point = mq.get_clipping()

            # evaluate
            acc = self.model.validate()

            del mq

            return point, loss, acc

        def eval_pnorm_on_calibration(p):
            self.qtype = 'lp_norm'
            self.lp = p
            mq = ModelQuantizer(self.model, layers, replacement_factory, **self.kwargs)
            loss = evaluate_calibration(mp)
            point = mq.get_clipping()

            del mq

            return point, loss

        ps = np.linspace(2, 4, 10)
        losses = []
        for p in tqdm(ps):
            point, loss = eval_pnorm_on_calibration(p)
            losses.append(loss.item())
            print("(p, loss) - ({}, {})".format(p, loss.item()))

        # Interpolate optimal p
        z = np.polyfit(ps, losses, 2)
        y = np.poly1d(z)
        p_intr = y.deriv().roots[0]
        # loss_opt = y(p_intr)
        print("p intr: {:.2f}".format(p_intr))

        lp_point, lp_loss, lp_acc = eval_pnorm(p_intr)

        print("loss p intr: {:.4f}".format(lp_loss.item()))
        print("acc p intr: {:.4f}".format(lp_acc))

        global _eval_count, _min_loss
        _min_loss = lp_loss.item()

        init = lp_point

        self.qtype = 'lp_norm'
        self.lp = p_intr

        mq = ModelQuantizer(self.model, layers, replacement_factory, **self.kwargs)

        # run optimizer
        min_options = {}
        if self.maxiter is not None:
            min_options['maxiter'] = self.maxiter
        if self.maxfev is not None:
            min_options['maxfev'] = self.maxfev

        _iter = count(0)

        def local_search_callback(x):
            it = next(_iter)
            mq.set_clipping(x, self.model.device)
            loss = evaluate_calibration(mp)
            print("\n[{}]: Local search callback".format(it))
            print("loss: {:.4f}\n".format(loss.item()))
            print(x)

            # evaluate
            acc = self.model.validate()

        self.min_method = "Powell"
        method = self.min_method
        res = opt.minimize(lambda scales: evaluate_calibration_clipped(scales, self.model, mq), init.cpu().numpy(),
                        method=method, options=min_options, callback=local_search_callback)

        print(res)

        scales = res.x
        mq.set_clipping(scales, self.model.device)
        loss = evaluate_calibration(mp)

        # evaluate
        acc = self.model.validate()
        data['powell'] = {'alpha': scales, 'loss': loss.item(), 'acc': acc}
