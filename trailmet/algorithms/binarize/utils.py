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
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
import torch.utils
from PIL import Image

__all__ = [
    'Lighting',
    'DistributionLoss',
    'adjust_learning_rate',
]

# lighting data augmentation
imagenet_pca = {
    'eigval':
    np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec':
    np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ]),
}


class Lighting(object):
    """
    Parameters
    ----------
        alphastd ():
        eigval (np.array): Eigen Value
        eigvec (np.array): Eigen Vector
    """

    def __init__(self,
                 alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3, )
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.0:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3, ))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class DistributionLoss(loss._Loss):
    """The KL-Divergence loss for the binary student model and real teacher
    output.

    output must be a pair of (model_output, real_output), both NxC tensors. The
    rows of real_output must all add up to one (probability scores); however,
    model_output must be the pre-softmax output of the network.

    Parameters
    ----------
        model_output (Tensor): Prediction from your model.
        real_output (Tensor): Ground Truth.
    """

    def forward(self, model_output, real_output):
        self.size_average = True

        if real_output.requires_grad:
            raise ValueError(
                'real network output should not require gradients.')

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        real_output_soft = F.softmax(real_output, dim=1)
        del model_output, real_output

        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        cross_entropy_loss = -torch.bmm(real_output_soft,
                                        model_output_log_prob)
        if self.size_average:
            cross_entropy_loss = cross_entropy_loss.mean()
        else:
            cross_entropy_loss = cross_entropy_loss.sum()
        return cross_entropy_loss


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30
    epochs."""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
