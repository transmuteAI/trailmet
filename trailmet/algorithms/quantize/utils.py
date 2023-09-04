# MIT License
#
# Copyright (c) 2023 Transmute AI Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import torch
from plotly import graph_objects

__all__ = [
    'get_qscheme',
    'get_dtype',
    'replace_activation_with_identity',
    'StopForwardException',
    'DataSaverHook',
    'GradSaverHook',
    'LinearTempDecay'
    'Node',
    'GraphPlotter'
]

def get_qscheme(per_channel=False, symmetric=False):
    if per_channel and symmetric:
        return torch.per_channel_symmetric
    elif per_channel and not symmetric:
        return torch.per_channel_affine
    elif not per_channel and symmetric:
        return torch.per_tensor_symmetric
    else:
        return torch.per_tensor_affine
    
def get_dtype(quant_min: int, quant_max: int):
    # bit capacity for qint and quint is reduced by 1 for 'x86' backend
    if quant_min>=0 and quant_max<=127:
        return torch.quint8
    elif quant_min>=-64 and quant_max<=63:
        return torch.qint8
    else:
        return torch.qint32

def replace_activation_with_identity(module: torch.nn.Module, activations: list) -> None:
    reassign = dict()
    for name, child_module in module.named_children():
        replace_activation_with_identity(child_module, activations)
        for activation in activations:
            if isinstance(child_module, activation):
                reassign[name] = torch.nn.Identity()
    for key, value in reassign.items():
        module._modules[key] = value

def quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.inp_quant(x)
    x = self._forward_impl(x)
    x = self.out_dequant(x)
    return x

class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph.
    """
    pass

class DataSaverHook:
    """
    Forward hook that stores the input and output of a layer.
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException
        
class GradSaverHook:
    """
    Backward hook that stores the gradients of a layer.
    """
    def __init__(self, store_grad=True):
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = None

    def __call__(self, module, grad_input, grad_output):
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException


class LinearTempDecay:
    """
    Class to implement a linear temperature decay scheduler for a given maximum time step.

    :param t_max: maximum number of time steps to decay temperature over.
    :param rel_start_decay: relative point in time to start the decay from the maximum time step. [default=.2]
    :param start_b: initial temperature value. [default=10]
    :param end_b: final temperature value. [default=2]

    """
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
        
class Node:
    def __init__(self, cost=0, profit=0, bit=None, parent=None, left=None, middle=None, right=None, position='middle'):
        self.parent = parent
        self.left = left
        self.middle = middle
        self.right = right
        self.position = position
        self.cost = cost
        self.profit = profit
        self.bit = bit

    def __str__(self):
        return 'cost: {:.2f} profit: {:.2f}'.format(self.cost, self.profit)
    
    def __repr__(self):
        return self.__str__()
    

class GraphPlotter:
    def __init__(self, save_dir: str = './'):
        self.save_dir = save_dir

    def line_plotter(self, columns, names, name_fmt: str = '{}', 
            title: str = '', xlabel: str = '', ylabel: str = '', ytype: str = '-'):
        data = [graph_objects.Scatter(
            y = columns[i], 
            mode = 'lines + markers',
            name = name_fmt.format(column_name),
        ) for i, column_name in enumerate(names)]
        layout = graph_objects.Layout(
            title = title,
            xaxis = dict(title=xlabel),
            yaxis = dict(title=ylabel, type=ytype)
        )
        fig = graph_objects.Figure(data, layout)
        self.save_plot(title, fig)

    def save_plot(self, title, fig: graph_objects.Figure):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        fig.write_image('{}/{}_plot.png'.format(self.save_dir, title))