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
from .chip import Chip
from .chipnet import ChipNet
from .growth_regularisation import Growth_Regularisation
from .network_slimming import Network_Slimming
from .pns import ChannelRounding, Conv2dWrapper, LinearWrapper, BN2dWrapper, SlimPruner
from .prune import BasePruning
from .pruner import pruner_dict
from .utils import (
    update_bn_grad,
    summary_model,
    is_depthwise_conv2d,
    prune_conv2d,
    prune_bn2d,
    prune_fc,
    cal_threshold_by_bn2d_weights,
    mask2idxes,
    top_k_idxes,
    ceil,
    round_up_to_power_of_2,
)
