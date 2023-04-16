# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import time
from collections import OrderedDict

import torch
import torch.nn as nn

from trailmet.models import resnet


# -

class IdentityModule(nn.Module):
    def forward(self, x):
        return x


# +
def convert_cifar_resnet(net, insert_identity_modules=False):
    """Convert a ResNetCifar module (in place)

    Returns
    -------
        net: the mutated net
    """
    
    net.conv1, net.bn1 = convert_conv_bn(net.conv1, net.bn1, torch.ones(3).byte(), get_gates(net.bn1))
    in_gates = torch.ones(net.conv1.out_channels).byte()

    clean_res = True
    net.layer1, in_gates = convert_layer(net.layer1, in_gates, insert_identity_modules, clean_res)
    net.layer2, in_gates = convert_layer(net.layer2, in_gates, insert_identity_modules, clean_res)
    net.layer3, in_gates = convert_layer(net.layer3, in_gates, insert_identity_modules, clean_res)
    net.fc = convert_fc_head(net.fc, in_gates)

    return net


def convert_layer(layer_module, in_gates, insert_identity_modules, clean_res):
    """Convert a ResnetCifar layer (in place)

    Parameters
    ----------
        layer_module: a nn.Sequential
        in_gates: mask

    Returns
    -------
        layer_module: mutated layer_module
        in_gates: ajusted mask
    """

    previous_layer_gates = in_gates

    new_blocks = []
    for block in layer_module:
        in_gates = convert_block(block, in_gates)
        new_blocks.append(IdentityModule())
        
    # Remove unused residual features
    if clean_res:
        print()
        cur_layer_gates = in_gates
        for block in new_blocks:
            if isinstance(block, IdentityModule):
                continue
            clean_block(previous_layer_gates, cur_layer_gates)  # in-place

    layer_module = nn.Sequential(*new_blocks)
    return layer_module, in_gates


def convert_block(block_module, in_gates):
    """Convert a Basic Resnet block (in place)

    Parameters
    ----------
        block_module: a BasicBlock
        in_gates: received mask

    Returns
    -------
        block_module: mutated block
        in_gates: out_gates of this block (in_gates for next block)
    """

    assert not hasattr(block_module, 'conv3')  # must be basic block

    b1_gates = get_gates(block_module.bn1)
    b2_gates = get_gates(block_module.bn2)

    delta_branch_is_pruned = b1_gates.sum().item() == 0 or b2_gates.sum().item() == 0
    
    # Delta branch
    if not delta_branch_is_pruned:
        block_module.conv1, block_module.bn1 = convert_conv_bn(block_module.conv1, block_module.bn1, in_gates, b1_gates)
        block_module.conv2, block_module.bn2 = convert_conv_bn(block_module.conv2, block_module.bn2, b1_gates, b2_gates)

    if block_module.downsample is not None:
        ds_gates = get_gates(block_module.downsample[1])
        ds_conv, ds_bn = convert_conv_bn(block_module.downsample[0], block_module.downsample[1], in_gates, ds_gates)
        ds_module = nn.Sequential(ds_conv, ds_bn)

        in_gates = elementwise_or(ds_gates, b2_gates)
    else:
        in_gates = elementwise_or(in_gates, b2_gates)

    return in_gates

def clean_block(previous_layer_alivef, cur_layer_alivef):
    """Remove unused res features (operates in-place)"""

    def clean_indices(idx, alive_mask=cur_layer_alivef):
        mask = i2mask(idx, alive_mask)
        mask = mask[mask2i(alive_mask)]
        return mask2i(mask)


def convert_conv_bn(conv_module, bn_module, in_gates, out_gates):
    in_indices = mask2i(in_gates)  # indices of kept features
    out_indices = mask2i(out_gates)

    # Keep the good ones
    new_conv_w = conv_module.weight.data[out_indices][:, in_indices]

    new_conv = make_conv(new_conv_w, from_module=conv_module)
    new_bn = convert_bn(bn_module, out_indices)

    new_conv.out_idx = out_indices
    
    return new_conv, new_bn


def convert_fc_head(fc_module, in_gates):
    """Convert a the final FC module of the net

    Parameters
    ----------
        fc_module: a nn.Linear with weight tensor of size (out_f, in_f)
        in_gates: binary vector or list of size in_f

    Returns
    -------
        fc_module: mutated module
    """

    in_indices = mask2i(in_gates)
    new_weight_tensor = fc_module.weight.data[:, in_indices]
    return make_fc(new_weight_tensor, from_module=fc_module)


def convert_bn(bn_module, out_indices):
    z = bn_module.get_gates(stochastic=False)
    new_weight = bn_module.weight.data[out_indices] * z[out_indices]
    new_bias = bn_module.bias.data[out_indices] * z[out_indices]

    new_bn_module = nn.BatchNorm2d(len(new_weight))
    new_bn_module.weight.data.copy_(new_weight)
    new_bn_module.bias.data.copy_(new_bias)
    new_bn_module.running_mean.copy_(bn_module.running_mean[out_indices])
    new_bn_module.running_var.copy_(bn_module.running_var[out_indices])

    new_bn_module.out_idx = out_indices

    return new_bn_module


def make_bn(bn_module, kept_indices):
    new_bn_module = nn.BatchNorm2d(len(kept_indices))
    new_bn_module.weight.data.copy_(bn_module.weight.data[kept_indices])
    new_bn_module.bias.data.copy_(bn_module.bias.data[kept_indices])
    new_bn_module.running_mean.copy_(bn_module.running_mean[kept_indices])
    new_bn_module.running_var.copy_(bn_module.running_var[kept_indices])

    if hasattr(bn_module, 'out_idx'):
        new_bn_module.out_idx = bn_module.out_idx[kept_indices]
    else:
        new_bn_module.out_idx = kept_indices

    return new_bn_module


def make_conv(weight_tensor, from_module):
    # NOTE: No bias

    # New weight size
    in_channels = weight_tensor.size(1)
    out_channels = weight_tensor.size(0)

    # Other params
    kernel_size = from_module.kernel_size
    stride = from_module.stride
    padding = from_module.padding

    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    conv.weight.data.copy_(weight_tensor)
    return conv


def make_fc(weight_tensor, from_module):
    in_features = weight_tensor.size(1)
    out_features = weight_tensor.size(0)
    fc = nn.Linear(in_features, out_features)
    fc.weight.data.copy_(weight_tensor)
    fc.bias.data.copy_(from_module.bias.data)
    return fc


def get_gates(module):
    #return module.get_gates(stochastic=False) > 0
    # Get the gates parameters by name
    gates_params = list(filter(lambda p: 'gate' in p[0], module.named_parameters()))

    # Extract the gates tensors from the parameters
    gates_tensors = [p[1] for p in gates_params]

    return gates_tensors

def elementwise_or(a, b):
    return (a + b) > 0


def mask2i(mask):
    assert mask.dtype == torch.uint8
    return mask.nonzero().view(-1)  # Note: do not use .squeeze() because single item becomes a scalar instead of 1-vec

def i2mask(i, from_tensor):
    x = torch.zeros_like(from_tensor)
    x[i] = 1
    return x
# -




