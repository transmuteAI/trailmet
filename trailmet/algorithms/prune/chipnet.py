# importing the required packages
from trailmet.algorithms.prune.prune import BasePruning
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from trailmet.utils import seed_everything

seed_everything(43)


class PrunableBatchNorm2d(torch.nn.BatchNorm2d):
    """wrapper class for batch norm layer to make it prunable"""
    def __init__(self, num_features: int, conv_module:nn.Conv2d = None):
        super(PrunableBatchNorm2d, self).__init__(num_features=num_features)
        self.is_imp = False
        self.is_pruned = False
        self.num_gates = num_features
        self.zeta = nn.Parameter(torch.rand(num_features) * 0.01)
        self.pruned_zeta = torch.ones_like(self.zeta)
        if conv_module is not None:
            def fo_hook(module, in_tensor, out_tensor):
                module.num_input_active_channels = (in_tensor[0].sum((0,2,3))>0).sum().item()
                module.output_area = out_tensor.size(2) * out_tensor.size(3)
            conv_module.register_forward_hook(fo_hook)
        self._conv_module = conv_module
        beta = 1.
        gamma = 2.
        for n, x in zip(('beta', 'gamma'), (torch.tensor([x], requires_grad=False) for x in (beta, gamma))):
            self.register_buffer(n, x)  # self.beta will be created (same for gamma, zeta)        

    def forward(self, input_data):
        out = super(PrunableBatchNorm2d, self).forward(input_data)
        z = self.pruned_zeta if self.is_pruned else self.get_zeta_t()
        out *= z[None, :, None, None] # broadcast the mask to all samples in the batch, and all locations
        return out

    def get_zeta_i(self):
        """returns the zeta_i by applying generalized logistic transformation on zeta"""
        return self.__generalized_logistic(self.zeta)

    def get_zeta_t(self):
        """returns zeta_t by applying continuous heaviside tranformation on zeta_i"""
        zeta_i = self.get_zeta_i()
        return self.__continuous_heaviside(zeta_i)

    def set_beta_gamma(self, beta, gamma):
        """sets the values of beta and gamma"""
        self.beta.data.copy_(torch.Tensor([beta]))
        self.gamma.data.copy_(torch.Tensor([gamma]))

    def __generalized_logistic(self, x):
        return 1./(1.+torch.exp(-self.beta*x))

    def __continuous_heaviside(self, x):
        return 1-torch.exp(-self.gamma*x)+x*torch.exp(-self.gamma)

    def prune(self, threshold):
        self.is_pruned = True
        self.pruned_zeta = (self.get_zeta_t()>threshold).float()
        self.zeta.requires_grad = False

    def unprune(self):
        self.is_pruned = False
        self.zeta.requires_grad = True

    def get_params_count(self):
        total_conv_params = self._conv_module.in_channels*self.pruned_zeta.shape[0]*self._conv_module.kernel_size[0]*self._conv_module.kernel_size[1]
        bn_params = self.num_gates*2
        active_bn_params = self.pruned_zeta.sum().item()*2
        active_conv_params = self._conv_module.num_input_active_channels*self.pruned_zeta.sum().item()*self._conv_module.kernel_size[0]*self._conv_module.kernel_size[1]
        return active_conv_params+active_bn_params, total_conv_params+bn_params

    def get_volume(self):
        total_volume = self._conv_module.output_area*self.num_gates
        active_volume = self._conv_module.output_area*self.pruned_zeta.sum().item()
        return active_volume, total_volume

    def get_flops(self):
        k_area = self._conv_module.kernel_size[0]*self._conv_module.kernel_size[1]
        total_flops = self._conv_module.output_area*self.num_gates*self._conv_module.in_channels*k_area
        active_flops = self._conv_module.output_area*self.pruned_zeta.sum().item()*self._conv_module.num_input_active_channels*k_area
        return active_flops, total_flops

    @staticmethod
    def from_batchnorm(bn_module, conv_module):
        new_bn = PrunableBatchNorm2d(bn_module.num_features, conv_module)
        return new_bn, conv_module


class ModuleInjection:
    pruning_method = 'full'
    prunable_modules = []

    @staticmethod
    def make_prunable(conv_module, bn_module):
        """Make a (conv, bn) sequence prunable.
        :param conv_module: A Conv2d module
        :param bn_module: The BatchNorm2d module following the Conv2d above
        :param prune_before_bn: Whether the pruning gates will be applied before or after the Batch Norm
        :return: a pair (conv, bn) that can be trained to
        """
        if ModuleInjection.pruning_method == 'full':
            return conv_module, bn_module
        new_bn, conv_module = PrunableBatchNorm2d.from_batchnorm(bn_module, conv_module=conv_module)
        ModuleInjection.prunable_modules.append(new_bn)
        return conv_module, new_bn

class ChipNet(BasePruning):
    """class to compress models using chipnet method"""
    def __init__(self, model, dataloaders, **kwargs):
        super(ChipNet, self).__init__(**kwargs)
        self.model = model
        self.dataloaders = dataloaders
        self.kwargs = kwargs
        self.budget_type = self.kwargs['CHIPNET_ARGS'].get('BUDGET_TYPE', 'channel_ratio')
        self.target_budget = self.kwargs['CHIPNET_ARGS'].get('TARGET_BUDGET', 0.5)
        self.steepness  = self.kwargs['CHIPNET_ARGS'].get('STEEPNESS', 10)
        self.budget_loss_weightage = self.kwargs['CHIPNET_ARGS'].get('BUDGET_LOSS_WEIGHTAGE', 30)
        self.crispness_loss_weightage = self.kwargs['CHIPNET_ARGS'].get('CRISPNESS_LOSS_WEIGHTAGE', 10)
        self.b_inc = self.kwargs['CHIPNET_ARGS'].get('BETA_INCREMENT', 5.)
        self.g_inc = self.kwargs['CHIPNET_ARGS'].get('GAMMA_INCREMENT', 2.)
        self.target_budget = torch.FloatTensor([self.target_budget]).to(self.device)
        self.steepness = 10
        self.ceLoss = nn.CrossEntropyLoss()

    def compress_model(self):
        """function to compress model using chipnet method."""
        self.model.to(self.device)

        if 'PRETRAIN' in self.kwargs:
            self.log_name = self.log_name + '_pretrained'
            self.base_train(self.model, self.dataloaders, **self.kwargs['PRETRAIN'])

        if 'PRUNE' in self.kwargs:
            self.log_name = self.log_name + '_pruning'
            print('preparing model for pruning')
            self.prepare_model_for_compression()
            self.model.to(self.device)
            self.prune(self.model, self.dataloaders, **self.kwargs['PRUNE'])

        if 'FINETUNE' in self.kwargs:
            self.prepare_for_finetuning(self.target_budget.item(), self.budget_type)
            self.log_name = self.log_name + '_finetuned'
            self.base_train(self.model, self.dataloaders, **self.kwargs['FINETUNE'])

    def prune(self, model, dataloaders, **kwargs):
        """function to prune a pretrained model using chipnet method"""
        num_epochs = kwargs.get('EPOCHS', 20)
        test_only = kwargs.get('TEST_ONLY', False)
        #### preparing optimizer ####
        lr = kwargs.get('LR', 0.001)
        weight_decay = kwargs.get('WEIGHT_DECAY', 0.001)
        param_optimizer = list(model.named_parameters())
        no_decay = ["zeta"]
        optimizer_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay,'lr':lr},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr':lr},
            ]
        optimizer = optim.AdamW(optimizer_parameters)
        #############################
        criterion = self.prune_criterion
        best_acc = 0
        beta, gamma = 1., 2.
        self.set_beta_gamma(beta, gamma)

        remaining_before_pruning = []
        remaining_after_pruning = []
        valid_accuracy = []
        pruning_accuracy = []
        pruning_threshold = []
        problems = []

        if test_only is False:
            for epoch in range(num_epochs):
                print(f'Starting epoch {epoch + 1} / {num_epochs}')
                self.unprune_model()
                self.train_one_epoch(model, dataloaders['train'], criterion, optimizer, self.steepness_update_function(5./len(dataloaders['train'])))
                print(f'[{epoch + 1} / {num_epochs}] Validation before pruning')
                acc, _ = self.test(model, dataloaders['val'], criterion)
                remaining = self.get_remaining(self.steepness, self.budget_type).item()
                remaining_before_pruning.append(remaining)
                valid_accuracy.append(acc)

                print(f'[{epoch + 1} / {num_epochs}] Validation after pruning')
                threshold, problem = self.prune_model(self.target_budget, self.budget_type)
                acc, _ = self.test(model, dataloaders['val'], criterion)
                remaining = self.get_remaining(self.steepness, self.budget_type).item()
                pruning_accuracy.append(acc)
                pruning_threshold.append(threshold)
                remaining_after_pruning.append(remaining)
                problems.append(problem)

                # 
                beta = min(6., beta + (0.1 / self.b_inc))
                gamma = min(256, gamma * (2**(1. / self.g_inc)))
                self.set_beta_gamma(beta, gamma)
                print("Changed beta to", beta, "changed gamma to", gamma)

                if acc > best_acc:
                    print("**Saving checkpoint**")
                    best_acc = acc
                    torch.save({
                        "epoch": epoch+1,
                        "beta": beta,
                        "gamma": gamma,
                        "prune_threshold": threshold,
                        "state_dict": model.state_dict(),
                        "accuracy": acc,
                    }, f"checkpoints/{self.log_name}.pth")

                df_data=np.array([remaining_before_pruning, remaining_after_pruning, valid_accuracy, pruning_accuracy, pruning_threshold, problems]).T
                df = pd.DataFrame(df_data,columns = ['Remaining before pruning', 'Remaining after pruning', 'Valid accuracy', 'Pruning accuracy', 'Pruning threshold', 'problems'])
                df.to_csv(f"logs/{self.log_name}.csv")

    def steepness_update_function(self, step):
        """returns function to update steepness in budget loss of chipnet"""
        def update():
            self.steepness = min(60, self.steepness+step) # increasing schedule of steepness to a maximum value of 60 to avoid gradient explosion.
        return update

    def prepare_model_for_compression(self):
        """prepares model for compression by replacing batchnorm layers"""
        ModuleInjection.pruning_method='prune'
        def replace_bn(m):
            for attr_str in dir(m):
                target_attr = getattr(m, attr_str)
                if type(target_attr) == torch.nn.BatchNorm2d:
                    conv_attr = getattr(m, attr_str.replace('bn','conv'))
                    conv, bn = ModuleInjection.make_prunable(conv_attr, target_attr)
                    setattr(m, attr_str.replace('bn','conv'), conv)
                    setattr(m, attr_str, bn)
            for _, ch in m.named_children():
                replace_bn(ch)
        self.prunable_modules = ModuleInjection.prunable_modules
        replace_bn(self.model)


    def prune_criterion(self, y_pred, y_true):
        """loss function for pruning"""
        ce_loss = self.ceLoss(y_pred, y_true)
        budget_loss = ((self.get_remaining(self.steepness, self.budget_type).to(self.device)-self.target_budget.to(self.device))**2).to(self.device)
        crispness_loss = self.get_crispnessLoss()
        return budget_loss*self.budget_loss_weightage + crispness_loss*self.crispness_loss_weightage + ce_loss

    def calculate_prune_threshold(self, target_budget, budget_type):
        """calculates the prune threshold for different budget types"""
        zetas = self.give_zetas()
        if budget_type in ['volume_ratio']:
            zeta_weights = self.give_zeta_weights()
            zeta_weights = zeta_weights[np.argsort(zetas)]
        zetas = sorted(zetas)
        if budget_type == 'volume_ratio':
            curr_budget = 0
            indx = 0
            while curr_budget < (1. - target_budget):
                indx += 1
                curr_budget += zeta_weights[indx]
            prune_threshold = zetas[indx]
        else:
            prune_threshold = zetas[int((1.-target_budget)*len(zetas))]
        return prune_threshold

    def smoothRound(self, x, steepness=20.):
        """function to apply smooth rounding on zeta for more accurate budget calculation"""
        return 1. / (1. + torch.exp(-1 * steepness*(x - 0.5)))

    def n_remaining(self, module, steepness=20.):
        """returns the remaining number of channels"""
        return (module.pruned_zeta if module.is_pruned else self.smoothRound(module.get_zeta_t(), steepness)).sum()

    def is_all_pruned(self, module):
        """checks if the whole block is pruned"""
        return self.n_remaining(module) == 0

    def get_remaining(self, steepness=20., budget_type = 'channel_ratio'):
        """return the fraction of active zeta_t (i.e > 0.5)"""
        n_rem = 0
        n_total = 0
        for l_block in self.prunable_modules:
            if budget_type == 'volume_ratio':
                n_rem += (self.n_remaining(l_block, steepness)*l_block._conv_module.output_area)
                n_total += (l_block.num_gates*l_block._conv_module.output_area)
            elif budget_type == 'channel_ratio':
                n_rem += self.n_remaining(l_block, steepness)
                n_total += l_block.num_gates
            elif budget_type == 'parameter_ratio':
                k = l_block._conv_module.kernel_size[0]
                prev_total = 3 if self.prev_module[l_block] is None else self.prev_module[l_block].num_gates
                prev_remaining = 3 if self.prev_module[l_block] is None else self.n_remaining(self.prev_module[l_block], steepness)
                n_rem += self.n_remaining(l_block, steepness)*prev_remaining*k*k
                n_total += l_block.num_gates*prev_total*k*k
            elif budget_type == 'flops_ratio':
                k = l_block._conv_module.kernel_size[0]
                output_area = l_block._conv_module.output_area
                prev_total = 3 if self.prev_module[l_block] is None else self.prev_module[l_block].num_gates
                prev_remaining = 3 if self.prev_module[l_block] is None else self.n_remaining(self.prev_module[l_block], steepness)
                curr_remaining = self.n_remaining(l_block, steepness)
                n_rem += curr_remaining*prev_remaining*k*k*output_area + curr_remaining*output_area
                n_total += l_block.num_gates*prev_total*k*k*output_area + l_block.num_gates*output_area
        return n_rem/n_total

    def give_zetas(self):
        """returns pruning gates as a list"""
        zetas = []
        for l_block in self.prunable_modules:
            zetas.append(l_block.get_zeta_t().cpu().detach().numpy().tolist())
        zetas = [z for k in zetas for z in k ]
        return zetas

    def give_zeta_weights(self):
        """returns the importance of pruning gates using the volume it represents, used for volume pruning"""
        zeta_weights = []
        for l_block in self.prunable_modules:
            zeta_weights.append([l_block._conv_module.output_area]*l_block.num_gates)
        zeta_weights = [z for k in zeta_weights for z in k ]
        return zeta_weights/np.sum(zeta_weights)

    def plot_zt(self):
        """plots the distribution of zeta_t and returns the same"""
        zetas = self.give_zetas()
        exactly_zeros = np.sum(np.array(zetas)==0.0)
        exactly_ones = np.sum(np.array(zetas)==1.0)
        plt.hist(zetas)
        plt.show()
        return exactly_zeros, exactly_ones

    def get_crispnessLoss(self):
        """loss responsible for making zeta_t 1 or 0"""
        loss = torch.FloatTensor([]).to(self.device)
        for l_block in self.prunable_modules:
            loss = torch.cat([loss, torch.pow(l_block.get_zeta_t()-l_block.get_zeta_i(), 2)])
        return torch.mean(loss).to(self.device)

    def prune_model(self, target_budget, budget_type='channel_ratio', finetuning=False, threshold=None):
        """prunes the network to make zeta_t exactly 1 and 0"""

        if budget_type == 'parameter_ratio':
            zetas = sorted(self.give_zetas())
            high = len(zetas)-1
            low = 0
            while low<high:
                mid = (high + low)//2
                threshold = zetas[mid]
                for l_block in self.prunable_modules:
                    l_block.prune(threshold)
                self.remove_orphans()
                if self.params() < target_budget:
                    high = mid-1
                else:
                    low = mid+1
        elif budget_type == 'flops_ratio':
            zetas = sorted(self.give_zetas())
            high = len(zetas)-1
            low = 0
            while low<high:
                mid = (high + low)//2
                threshold = zetas[mid]
                for l_block in self.prunable_modules:
                    l_block.prune(threshold)
                self.remove_orphans()
                if self.flops() < target_budget:
                    high = mid-1
                else:
                    low = mid+1
        else:
            if threshold==None:
                self.prune_threshold = self.calculate_prune_threshold(target_budget, budget_type)
                threshold = min(self.prune_threshold, 0.9)

        for l_block in self.prunable_modules:
            l_block.prune(threshold)

        if finetuning:
            self.remove_orphans()
            return threshold
        else:
            problem = self.check_abnormality()
            return threshold, problem

    def unprune_model(self):
        """unprunes the network to again make pruning gates continuous"""
        for l_block in self.prunable_modules:
            l_block.unprune()

    def prepare_for_finetuning(self, budget, budget_type = 'channel_ratio'):
        """freezes zeta"""
        self.model(torch.rand(2, 3, 32, 32).to(self.device))
        threshold = self.prune_model(budget, budget_type=budget_type, finetuning=True)
        if budget_type not in ['parameter_ratio', 'flops_ratio']:
            while self.get_remaining(steepness=20., budget_type=budget_type)<budget:
                threshold -= 0.0001
                self.prune_model(budget, finetuning=True, budget_type=budget_type, threshold=threshold)
        return threshold

    def get_params_count(self):
        """returns the number of active and total parameters in the network"""
        total_params = 0
        active_params = 0
        for l_block in self.model.modules():
            if isinstance(l_block, PrunableBatchNorm2d):
                active_param, total_param = l_block.get_params_count()
                active_params += active_param
                total_params += total_param
            if isinstance(l_block, nn.Linear):
                linear_params = l_block.weight.view(-1).shape[0]
                active_params += linear_params
                total_params += linear_params
        return active_params, total_params

    def get_volume(self):
        """returns the active and total volume of the network"""
        total_volume = 0.
        active_volume = 0.
        for l_block in self.prunable_modules:
            active_volume_, total_volume_ = l_block.get_volume()
            active_volume += active_volume_
            total_volume += total_volume_
        return active_volume, total_volume

    def get_flops(self):
        """returns the active and total flops of the network"""
        total_flops = 0.
        active_flops = 0.
        for l_block in self.prunable_modules:
            active_flops_, total_flops_ = l_block.get_flops()
            active_flops += active_flops_
            total_flops += total_flops_
        return active_flops, total_flops

    def get_channels(self):
        """returns the active and total number of channels in the network"""
        total_channels = 0.
        active_channels = 0.
        for l_block in self.prunable_modules:
            active_channels+=l_block.pruned_zeta.sum().item()
            total_channels+=l_block.num_gates
        return active_channels, total_channels

    def set_beta_gamma(self, beta, gamma):
        """explicitly sets the values of beta and gamma parameters"""
        for l_block in self.prunable_modules:
            l_block.set_beta_gamma(beta, gamma)

    def check_abnormality(self):
        """checks for any abnormality in the pruning process"""
        n_removable = self.removable_orphans()
        isbroken = self.check_if_broken()
        if n_removable!=0. and isbroken:
            return f'both rem_{n_removable} and broken'
        if n_removable!=0.:
            return f'removable_{n_removable}'
        if isbroken:
            return 'broken'

    def check_if_broken(self):
        """checks if the network is broken due to abnormal pruning"""
        for bn in self.prunable_modules:
            if bn.is_imp and bn.pruned_zeta.sum() == 0:
                return True
        return False

    def removable_orphans(self):
        """checks for orphan nodes from the network that are not being used in the computation graph"""
        num_removed = 0
        bn_layers = self.model.get_bn_layers()
        for l_blocks in bn_layers:
            flag = sum([self.is_all_pruned(m) for m in l_blocks])
            if flag:
                num_removed += sum([self.n_remaining(m) for m in l_blocks])
        return num_removed

    def remove_orphans(self):
        """removes orphan nodes from the network that are not being used in the computation graph"""
        num_removed = 0
        bn_layers = self.model.get_bn_layers()
        for l_blocks in bn_layers:
            flag = sum([self.is_all_pruned(m) for m in l_blocks])
            if flag:
                num_removed += sum([self.n_remaining(m) for m in l_blocks])
                for m in l_blocks:
                    m.pruned_zeta.data.copy_(torch.zeros_like(m.pruned_zeta))
        return num_removed