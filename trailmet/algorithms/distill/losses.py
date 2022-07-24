import torch
import torch.nn as nn
import torch.nn.functional as F

class KDTransferLoss(nn.Module):
    def __init__(self,temperature):
        super().__init__()
        self.temperature=temperature
        self.kd_fun = nn.KLDivLoss(reduction='batchmean')
    def forward(self,out_t,out_s):
        s_max = F.log_softmax(out_s / self.temperature, dim=1)
        t_max = F.softmax(out_t / self.temperature, dim=1)
        loss_kd = self.kd_fun(s_max, t_max)
        return loss_kd