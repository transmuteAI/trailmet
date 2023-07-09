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
import torch
import torch.nn as nn
import torch.nn.functional as F
from trailmet.utils.functions import pdist

__all__ = ['KDTransferLoss', 'RkdDistance', 'RKdAngle']


class KDTransferLoss(nn.Module):
    """Class for KD Transfer Loss."""

    def __init__(self, temperature, reduction):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.kd_fun = nn.KLDivLoss(reduction=self.reduction)

    def forward(self, out_t, out_s):
        s_max = F.log_softmax(out_s / self.temperature, dim=1)
        t_max = F.softmax(out_t / self.temperature, dim=1)
        loss_kd = self.kd_fun(s_max, t_max)
        return loss_kd


class RkdDistance(nn.Module):
    """Class for RKD Distance."""

    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss


class RKdAngle(nn.Module):
    """Class for RKD Angle."""

    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = teacher.unsqueeze(0) - teacher.unsqueeze(1)
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = student.unsqueeze(0) - student.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss
