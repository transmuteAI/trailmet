
import torch
import torch.nn as nn
import numpy as np
from trailmet.algorithms.quantize.quantize import Conv2dFunctor, LinearFunctor
from trailmet.algorithms.quantize.methods import LearnedStepSizeQuantization, FixedClipValueQuantization
from trailmet.algorithms.quantize.methods import MaxAbsStaticQuantization, LpNormQuantization


quantization_mapping = {
    'max_static' : MaxAbsStaticQuantization,
    'lp_norm' : LpNormQuantization
}

def is_positive(module):
    return isinstance(module, nn.ReLU) or isinstance(module, nn.ReLU6)


class ActivationModuleWrapper(nn.Module):
    def __init__(self, name, wrapped_module, **kwargs):
        super(ActivationModuleWrapper, self).__init__()
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
                self.out_quantization_default = quantization_mapping[self.qtype](
                    self, tensor, self.bits_out,
                    symmetric=(not is_positive(wrapped_module)),
                    uint=True, kwargs=kwargs
                )
                self.out_quantization = self.out_quantization_default
                # print("ActivationModuleWrapperPost - {} | {} | {}".format(self.name, str(self.out_quantization), str(tensor.device)))

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

    @staticmethod
    def verify_initialized(quantization_handle, tensor, init_fn):
        if quantization_handle is None:
            init_fn(tensor)    
    
    def get_quantization(self):
        return self.out_quantization

    def set_quantization(self, qtype, kwargs):
        self.out_quantization = qtype(
            self, self.bits_out, 
            symmetric=(not is_positive(self.wrapped_module)),
            uint=True, kwargs=kwargs
        )


class ParameterModuleWrapper(nn.Module):
    def __init__(self, name, wrapped_module, **kwargs):
        super(ParameterModuleWrapper, self).__init__()
        self.name = name
        self.wrapped_module = wrapped_module
        self.forward_functor = kwargs['forward_functor']
        self.bit_weights = kwargs['bit_weights']
        self.bits_out = kwargs['bits_out']
        self.qtype = kwargs['qtype']
        self.bcorr_w = kwargs['bcorr_w']
        self.bn = kwargs['bn'] if 'bn' in kwargs else None
        self.enabled = True
        self.active = True
        self.centroids_hist = {}
        self.log_weights_hist = False
        self.log_weights_mse = False
        self.log_clustering = False
        self.dynamic_weight_quantization = True

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
            # print("ParameterModuleWrapperPost - {} | {} | {}".format(self.name, str(self.weight_quantization),str(self.weight.device)))

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

    def set_quantization(self, qtype, kwargs):
        self.weight_quantization = qtype(self, self.bit_weights, symmetric=True, uint=True, kwargs=kwargs)
    
                                                

class ModelQuantizer:
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
    
