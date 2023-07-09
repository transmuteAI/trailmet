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
# importing the required packages
from trailmet.algorithms.prune.prune import BasePruning
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

import logging
from datetime import datetime
from tqdm import tqdm
import wandb
import pandas as pd
import numpy as np
import os
import time

from trailmet.utils import (
    AverageMeter,
    accuracy,
    save_checkpoint,
    seed_everything,
    adjust_learning_rate,
)

logger = logging.getLogger(__name__)

seed_everything(43)


class PrunableBatchNorm2d(torch.nn.BatchNorm2d):
    """Wrapper class for batch norm layer to make it prunable."""

    def __init__(self, num_features: int, conv_module: nn.Conv2d = None):
        super(PrunableBatchNorm2d, self).__init__(num_features=num_features)
        self.is_imp = False
        self.is_pruned = False
        self.num_gates = num_features
        self.zeta = nn.Parameter(torch.rand(num_features) * 0.01)
        self.pruned_zeta = torch.ones_like(self.zeta)
        if conv_module is not None:

            def fo_hook(module, in_tensor, out_tensor):
                module.num_input_active_channels = ((in_tensor[0].sum(
                    (0, 2, 3)) > 0).sum().item())
                module.output_area = out_tensor.size(2) * out_tensor.size(3)

            conv_module.register_forward_hook(fo_hook)
        self._conv_module = conv_module
        beta = 1.0
        gamma = 2.0
        for n, x in zip(
            ('beta', 'gamma'),
            (torch.tensor([x], requires_grad=False) for x in (beta, gamma)),
        ):
            self.register_buffer(
                n, x)  # self.beta will be created (same for gamma, zeta)

    def forward(self, input_data):
        out = super(PrunableBatchNorm2d, self).forward(input_data)
        z = self.pruned_zeta if self.is_pruned else self.get_zeta_t()
        out *= z[
            None, :, None,
            None]  # broadcast the mask to all samples in the batch, and all locations
        return out

    def get_zeta_i(self):
        """Returns the zeta_i by applying generalized logistic transformation
        on zeta."""
        return self.__generalized_logistic(self.zeta)

    def get_zeta_t(self):
        """Returns zeta_t by applying continuous heaviside transformation on
        zeta_i."""
        zeta_i = self.get_zeta_i()
        return self.__continuous_heaviside(zeta_i)

    def set_beta_gamma(self, beta, gamma):
        """Sets the values of beta and gamma."""
        self.beta.data.copy_(torch.Tensor([beta]))
        self.gamma.data.copy_(torch.Tensor([gamma]))

    def __generalized_logistic(self, x):
        return 1.0 / (1.0 + torch.exp(-self.beta * x))

    def __continuous_heaviside(self, x):
        return 1 - torch.exp(-self.gamma * x) + x * torch.exp(-self.gamma)

    def prune(self, threshold):
        self.is_pruned = True
        self.pruned_zeta = (self.get_zeta_t() > threshold).float()
        self.zeta.requires_grad = False

    def unprune(self):
        self.is_pruned = False
        self.zeta.requires_grad = True

    def get_params_count(self):
        total_conv_params = (self._conv_module.in_channels *
                             self.pruned_zeta.shape[0] *
                             self._conv_module.kernel_size[0] *
                             self._conv_module.kernel_size[1])
        bn_params = self.num_gates * 2
        active_bn_params = self.pruned_zeta.sum().item() * 2
        active_conv_params = (self._conv_module.num_input_active_channels *
                              self.pruned_zeta.sum().item() *
                              self._conv_module.kernel_size[0] *
                              self._conv_module.kernel_size[1])
        return active_conv_params + active_bn_params, total_conv_params + bn_params

    def get_volume(self):
        total_volume = self._conv_module.output_area * self.num_gates
        active_volume = self._conv_module.output_area * self.pruned_zeta.sum(
        ).item()
        return active_volume, total_volume

    def get_flops(self):
        k_area = self._conv_module.kernel_size[
            0] * self._conv_module.kernel_size[1]
        total_flops = (self._conv_module.output_area * self.num_gates *
                       self._conv_module.in_channels * k_area)
        active_flops = (self._conv_module.output_area *
                        self.pruned_zeta.sum().item() *
                        self._conv_module.num_input_active_channels * k_area)
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
        :param prune_before_bn: Whether the pruning gates will be applied
            before or after the Batch Norm
        :return: a pair (conv, bn) that can be trained to
        """
        if ModuleInjection.pruning_method == 'full':
            return conv_module, bn_module
        new_bn, conv_module = PrunableBatchNorm2d.from_batchnorm(
            bn_module, conv_module=conv_module)
        ModuleInjection.prunable_modules.append(new_bn)
        return conv_module, new_bn


class ChipNet(BasePruning):
    """Class to compress models using chipnet method.

    Parameters
    ----------
        model (object): A pytorch model you want to use.
        dataloaders (dict): Dictionary with dataloaders for train, val and test. Keys: 'train', 'val', 'test'.
        kwargs (object): YAML safe loaded file with information like chipnet_args(budget_type, channel_ratio, etc.)
    """

    def __init__(self, model, dataloaders, **kwargs):
        super(ChipNet, self).__init__(**kwargs)
        self.model = model
        self.dataloaders = dataloaders
        self.kwargs = kwargs
        self.budget_type = self.kwargs['CHIPNET_ARGS'].get(
            'BUDGET_TYPE', 'channel_ratio')
        self.target_budget = self.kwargs['CHIPNET_ARGS'].get(
            'TARGET_BUDGET', 0.5)
        self.steepness = self.kwargs['CHIPNET_ARGS'].get('STEEPNESS', 10)
        self.budget_loss_weightage = self.kwargs['CHIPNET_ARGS'].get(
            'BUDGET_LOSS_WEIGHTAGE', 30)
        self.crispness_loss_weightage = self.kwargs['CHIPNET_ARGS'].get(
            'CRISPNESS_LOSS_WEIGHTAGE', 10)
        self.b_inc = self.kwargs['CHIPNET_ARGS'].get('BETA_INCREMENT', 5.0)
        self.g_inc = self.kwargs['CHIPNET_ARGS'].get('GAMMA_INCREMENT', 2.0)
        self.target_budget = torch.FloatTensor([self.target_budget
                                                ]).to(self.device)
        self.steepness = 10
        self.ceLoss = nn.CrossEntropyLoss()

        self.wandb_monitor = self.kwargs['CHIPNET_ARGS'].get('WANDB', 'False')
        self.dataset_name = dataloaders['train'].dataset.__class__.__name__
        self.save = './checkpoints/'

        self.name = '_'.join([self.dataset_name, 'ChipNet'])

        os.makedirs(f'{os.getcwd()}/logs/ChipNet', exist_ok=True)
        os.makedirs(self.save, exist_ok=True)

        if self.wandb_monitor:
            wandb.init(project='Trailmet ChipNet', name=self.name)
            wandb.config.update(self.kwargs)

    def compress_model(self):
        """Function to compress model using chipnet method."""
        self.model.to(self.device)

        if 'PRETRAIN' in self.kwargs:
            print('Pretrainning the model')
            self.kwargs['PRETRAIN']['phase'] = 'PRETRAIN'
            self.base_train(self.model, self.dataloaders,
                            **self.kwargs['PRETRAIN'])

        if 'PRUNE' in self.kwargs:
            print('preparing model for pruning')
            self.kwargs['PRUNE']['phase'] = 'PRUNE'
            self.prepare_model_for_compression()
            self.model.to(self.device)
            self.prune(self.model, self.dataloaders, **self.kwargs['PRUNE'])

        if 'FINETUNE' in self.kwargs:
            print('Finetuning the model')
            self.kwargs['FINETUNE']['phase'] = 'FINETUNE'
            self.prepare_for_finetuning(self.target_budget.item(),
                                        self.budget_type)
            self.base_train(self.model, self.dataloaders,
                            **self.kwargs['FINETUNE'])

    def base_train(self, model, dataloaders, **kwargs) -> None:
        """This function is used to perform standard model training and can be
        used for various purposes, such as model pretraining, fine-tuning of
        compressed models, among others.

        For cases, where base_train is not directly applicable, feel free to
        overwrite wherever this parent class is inherited.
        """
        best_top1_acc = 0  # setting to lowest possible value
        num_epochs = kwargs.get('EPOCHS', self.pretraining_epochs)
        test_only = kwargs.get('TEST_ONLY', False)

        ### preparing optimizer ###
        optimizer_name = kwargs.get('OPTIMIZER', 'SGD')
        lr = kwargs.get('LR', 0.05)
        scheduler_type = kwargs.get('SCHEDULER_TYPE', 1)
        weight_decay = kwargs.get('WEIGHT_DECAY', 0.001)

        optimizer = self.get_optimizer(optimizer_name, model, lr, weight_decay)

        criterion = nn.CrossEntropyLoss()

        phase = kwargs['phase']
        self.name = '_'.join([
            self.dataset_name,
            f'{num_epochs}',
            f'{lr}',
            datetime.now().strftime('%b-%d_%H:%M:%S'),
        ])

        self.logger_file = f'{os.getcwd()}/logs/ChipNet/{self.name}.log'

        logging.basicConfig(
            filename=self.logger_file,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )

        logger.info(f'{phase} Arguments: {self.kwargs}')

        if test_only is False:
            epochs_list = []
            val_top1_acc_list = []
            val_top5_acc_list = []
            for epoch in range(num_epochs):
                adjust_learning_rate(optimizer, epoch, num_epochs,
                                     scheduler_type, lr)

                # t_loss = self.train_one_epoch(model, dataloaders['train'], criterion, optimizer)
                # acc, v_loss = self.test(model, dataloaders['val'], criterion)

                t_loss = self.train_one_epoch(
                    model,
                    dataloaders['train'],
                    criterion,
                    optimizer,
                    epoch,
                    phase,
                )

                valid_loss, valid_top1_acc, valid_top5_acc = self.test(
                    model, dataloaders['val'], criterion, epoch, phase)

                is_best = False
                if valid_top1_acc > best_top1_acc:
                    best_top1_acc = valid_top1_acc
                    is_best = True

                save_checkpoint(
                    {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_top1_acc': best_top1_acc,
                        'optimizer': optimizer.state_dict(),
                    },
                    is_best,
                    self.save,
                )

                if self.wandb_monitor:
                    wandb.log({'best_top1_acc': best_top1_acc})

                epochs_list.append(epoch)
                val_top1_acc_list.append(valid_top1_acc.cpu().numpy())
                val_top5_acc_list.append(valid_top5_acc.cpu().numpy())

                df_data = np.array([
                    epochs_list,
                    val_top1_acc_list,
                    val_top5_acc_list,
                ]).T
                df = pd.DataFrame(
                    df_data,
                    columns=[
                        'Epochs',
                        'Validation Top1',
                        'Validation Top5',
                    ],
                )
                df.to_csv(
                    f'{os.getcwd()}/logs/ChipNet/{phase}_{self.name}.csv',
                    index=False,
                )

        state = torch.load(f'{self.save}/model_best.pth.tar')
        model.load_state_dict(state['state_dict'], strict=True)
        valid_loss, valid_top1_acc, valid_top5_acc = self.test(
            model, dataloaders['test'], criterion, epoch, phase)
        print(
            f"Test Accuracy: {valid_top1_acc} | Valid Accuracy: {state['best_top1_acc']}"
        )

    def train_one_epoch(
        self,
        model,
        dataloader,
        loss_fn,
        optimizer,
        epoch,
        phase,
        extra_functionality=None,
    ):
        """Standard training loop which can be used for various purposes with
        an extra functionality function to add to its working at the end of the
        loop."""
        model.train()

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        end = time.time()

        epoch_iterator = tqdm(
            dataloader,
            desc=
            'Training X Epoch [X] (X / X Steps) (batch time=X.Xs) (data time=X.Xs) (loss=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        for i, (images, labels) in enumerate(epoch_iterator):
            data_time.update(time.time() - end)
            images = images.to(device=self.device)
            labels = labels.to(device=self.device)
            scores = model(images)

            loss = loss_fn(scores, labels)
            n = images.size(0)
            losses.update(loss.item(), n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            epoch_iterator.set_description(
                'Training %s Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f)'
                % (
                    phase,
                    epoch,
                    (i + 1),
                    len(dataloader),
                    batch_time.val,
                    data_time.val,
                    losses.val,
                ))

            logger.info(
                'Training %s Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (data time=%2.5fs) (loss=%2.5f)'
                % (
                    phase,
                    epoch,
                    (i + 1),
                    len(dataloader),
                    batch_time.val,
                    data_time.val,
                    losses.val,
                ))

            if self.wandb_monitor:
                wandb.log({
                    f'{phase}_train_loss': losses.val,
                })

            if extra_functionality is not None:
                extra_functionality()

        return losses.avg

    def test(self, model, dataloader, loss_fn=None, epoch=0, phase='PRETRAIN'):
        """This method is used to test the performance of the trained model."""

        model.eval()

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        epoch_iterator = tqdm(
            dataloader,
            desc=
            'Validating X Epoch [X] (X / X Steps) (batch time=X.Xs) (loss=X.X) (top1=X.X) (top5=X.X)',
            bar_format='{l_bar}{r_bar}',
            dynamic_ncols=True,
            disable=False,
        )

        with torch.no_grad():
            end = time.time()
            for i, (images, targets) in enumerate(epoch_iterator):
                images = images.to(self.device)
                targets = targets.to(self.device)
                outputs = model(images)
                pred1, pred5 = accuracy(outputs, targets, topk=(1, 5))

                if loss_fn is not None:
                    loss = loss_fn(outputs, targets)

                n = images.size(0)
                losses.update(loss.item(), n)
                top1.update(pred1[0], n)
                top5.update(pred5[0], n)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                epoch_iterator.set_description(
                    'Validating %s Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
                    % (
                        phase,
                        epoch,
                        (i + 1),
                        len(dataloader),
                        batch_time.val,
                        losses.val,
                        top1.val,
                        top5.val,
                    ))

                logger.info(
                    'Validating %s Epoch [%d] (%d / %d Steps) (batch time=%2.5fs) (loss=%2.5f) (top1=%2.5f) (top5=%2.5f)'
                    % (
                        phase,
                        epoch,
                        (i + 1),
                        len(dataloader),
                        batch_time.val,
                        losses.val,
                        top1.val,
                        top5.val,
                    ))

                if self.wandb_monitor:
                    wandb.log({
                        f'{phase}_val_loss': losses.val,
                        f'{phase}_val_top1_acc': top1.val,
                        f'{phase}_val_top5_acc': top5.val,
                    })

            print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'.format(
                top1=top1, top5=top5))

        return losses.avg, top1.avg, top5.avg

    def prune(self, model, dataloaders, **kwargs):
        """Function to prune a pretrained model using chipnet method."""
        num_epochs = kwargs.get('EPOCHS', 20)
        test_only = kwargs.get('TEST_ONLY', False)
        #### preparing optimizer ####
        lr = kwargs.get('LR', 0.001)
        weight_decay = kwargs.get('WEIGHT_DECAY', 0.001)
        param_optimizer = list(model.named_parameters())
        no_decay = ['zeta']
        optimizer_parameters = [
            {
                'params': [
                    p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                weight_decay,
                'lr':
                lr,
            },
            {
                'params': [
                    p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.0,
                'lr':
                lr,
            },
        ]

        optimizer = optim.AdamW(optimizer_parameters)
        phase = kwargs['phase']
        criterion = self.prune_criterion
        best_acc = 0
        beta, gamma = 1.0, 2.0
        self.set_beta_gamma(beta, gamma)

        remaining_before_pruning = []
        remaining_after_pruning = []
        valid_accuracy = []
        pruning_accuracy = []
        pruning_threshold = []
        problems = []

        if test_only is False:
            for epoch in range(num_epochs):
                print(f'Starting epoch {epoch} / {num_epochs}')
                self.unprune_model()
                self.train_one_epoch(
                    model,
                    dataloaders['train'],
                    criterion,
                    optimizer,
                    epoch,
                    phase,
                    extra_functionality=self.steepness_update_function(
                        5.0 / len(dataloaders['train'])),
                )
                print(
                    f'[{epoch + 1} / {num_epochs}] Validation before pruning')
                logger.info(
                    f'[{epoch + 1} / {num_epochs}] Validation before pruning')
                _, acc, _ = self.test(model, dataloaders['val'], criterion,
                                      epoch, phase)
                remaining = self.get_remaining(self.steepness,
                                               self.budget_type).item()
                remaining_before_pruning.append(remaining)
                valid_accuracy.append(acc.cpu().numpy())

                print(f'[{epoch + 1} / {num_epochs}] Validation after pruning')
                logger.info(
                    f'[{epoch + 1} / {num_epochs}] Validation after pruning')
                threshold, problem = self.prune_model(self.target_budget,
                                                      self.budget_type)
                _, acc, _ = self.test(model, dataloaders['val'], criterion,
                                      epoch, phase)
                remaining = self.get_remaining(self.steepness,
                                               self.budget_type).item()
                pruning_accuracy.append(acc.cpu().numpy())
                pruning_threshold.append(threshold)
                remaining_after_pruning.append(remaining)
                problems.append(problem)

                beta = min(6.0, beta + (0.1 / self.b_inc))
                gamma = min(256, gamma * (2**(1.0 / self.g_inc)))
                self.set_beta_gamma(beta, gamma)
                print('Changed beta to', beta, 'changed gamma to', gamma)

                if acc > best_acc:
                    print('**Saving checkpoint**')
                    best_acc = acc
                    torch.save(
                        {
                            'epoch': epoch + 1,
                            'beta': beta,
                            'gamma': gamma,
                            'prune_threshold': threshold,
                            'state_dict': model.state_dict(),
                            'accuracy': acc,
                        },
                        f'checkpoints/{self.log_name}.pth',
                    )

                is_best = False
                if acc > best_acc:
                    best_acc = acc
                    is_best = True

                save_checkpoint(
                    {
                        'epoch': epoch,
                        'beta': beta,
                        'gamma': gamma,
                        'prune_threshold': threshold,
                        'state_dict': model.state_dict(),
                        'accuracy': acc,
                        'optimizer': optimizer.state_dict(),
                    },
                    is_best,
                    self.save,
                )

                df_data = np.array([
                    remaining_before_pruning,
                    remaining_after_pruning,
                    valid_accuracy,
                    pruning_accuracy,
                    pruning_threshold,
                    problems,
                ]).T
                df = pd.DataFrame(
                    df_data,
                    columns=[
                        'Remaining before pruning',
                        'Remaining after pruning',
                        'Valid accuracy',
                        'Pruning accuracy',
                        'Pruning threshold',
                        'problems',
                    ],
                )

                df.to_csv(
                    f'{os.getcwd()}/logs/ChipNet/{phase}_{self.name}.csv',
                    index=False,
                )

    def steepness_update_function(self, step):
        """Returns function to update steepness in budget loss of chipnet."""

        def update():
            self.steepness = min(
                60, self.steepness + step
            )  # increasing schedule of steepness to a maximum value of 60 to avoid gradient explosion.

        return update

    def prepare_model_for_compression(self):
        """Prepares model for compression by replacing batchnorm layers."""
        ModuleInjection.pruning_method = 'prune'

        def replace_bn(m):
            for attr_str in dir(m):
                target_attr = getattr(m, attr_str)
                if type(target_attr) == torch.nn.BatchNorm2d:
                    conv_attr = getattr(m, attr_str.replace('bn', 'conv'))
                    conv, bn = ModuleInjection.make_prunable(
                        conv_attr, target_attr)
                    setattr(m, attr_str.replace('bn', 'conv'), conv)
                    setattr(m, attr_str, bn)
            for _, ch in m.named_children():
                replace_bn(ch)

        self.prunable_modules = ModuleInjection.prunable_modules
        replace_bn(self.model)

    def prune_criterion(self, y_pred, y_true):
        """Loss function for pruning."""
        ce_loss = self.ceLoss(y_pred, y_true)
        budget_loss = ((self.get_remaining(self.steepness,
                                           self.budget_type).to(self.device) -
                        self.target_budget.to(self.device))**2).to(self.device)
        crispness_loss = self.get_crispnessLoss()
        return (budget_loss * self.budget_loss_weightage +
                crispness_loss * self.crispness_loss_weightage + ce_loss)

    def calculate_prune_threshold(self, target_budget, budget_type):
        """Calculates the prune threshold for different budget types."""
        zetas = self.give_zetas()
        if budget_type in ['volume_ratio']:
            zeta_weights = self.give_zeta_weights()
            zeta_weights = zeta_weights[np.argsort(zetas)]
        zetas = sorted(zetas)
        if budget_type == 'volume_ratio':
            curr_budget = 0
            indx = 0
            while curr_budget < (1.0 - target_budget):
                indx += 1
                curr_budget += zeta_weights[indx]
            prune_threshold = zetas[indx]
        else:
            prune_threshold = zetas[int((1.0 - target_budget) * len(zetas))]
        return prune_threshold

    def smoothRound(self, x, steepness=20.0):
        """Function to apply smooth rounding on zeta for more accurate budget
        calculation."""
        return 1.0 / (1.0 + torch.exp(-1 * steepness * (x - 0.5)))

    def n_remaining(self, module, steepness=20.0):
        """Returns the remaining number of channels."""
        return (module.pruned_zeta if module.is_pruned else self.smoothRound(
            module.get_zeta_t(), steepness)).sum()

    def is_all_pruned(self, module):
        """Checks if the whole block is pruned."""
        return self.n_remaining(module) == 0

    def get_remaining(self, steepness=20.0, budget_type='channel_ratio'):
        """Return the fraction of active zeta_t (i.e > 0.5)"""
        n_rem = 0
        n_total = 0
        for l_block in self.prunable_modules:
            if budget_type == 'volume_ratio':
                n_rem += (self.n_remaining(l_block, steepness) *
                          l_block._conv_module.output_area)
                n_total += l_block.num_gates * l_block._conv_module.output_area
            elif budget_type == 'channel_ratio':
                n_rem += self.n_remaining(l_block, steepness)
                n_total += l_block.num_gates
            elif budget_type == 'parameter_ratio':
                k = l_block._conv_module.kernel_size[0]
                prev_total = (3 if self.prev_module[l_block] is None else
                              self.prev_module[l_block].num_gates)
                prev_remaining = (3 if self.prev_module[l_block] is None
                                  else self.n_remaining(
                                      self.prev_module[l_block], steepness))
                n_rem += self.n_remaining(l_block,
                                          steepness) * prev_remaining * k * k
                n_total += l_block.num_gates * prev_total * k * k
            elif budget_type == 'flops_ratio':
                k = l_block._conv_module.kernel_size[0]
                output_area = l_block._conv_module.output_area
                prev_total = (3 if self.prev_module[l_block] is None else
                              self.prev_module[l_block].num_gates)
                prev_remaining = (3 if self.prev_module[l_block] is None
                                  else self.n_remaining(
                                      self.prev_module[l_block], steepness))
                curr_remaining = self.n_remaining(l_block, steepness)
                n_rem += (
                    curr_remaining * prev_remaining * k * k * output_area +
                    curr_remaining * output_area)
                n_total += (
                    l_block.num_gates * prev_total * k * k * output_area +
                    l_block.num_gates * output_area)
        return n_rem / n_total

    def give_zetas(self):
        """Returns pruning gates as a list."""
        zetas = []
        for l_block in self.prunable_modules:
            zetas.append(l_block.get_zeta_t().cpu().detach().numpy().tolist())
        zetas = [z for k in zetas for z in k]
        return zetas

    def give_zeta_weights(self):
        """Returns the importance of pruning gates using the volume it
        represents, used for volume pruning."""
        zeta_weights = []
        for l_block in self.prunable_modules:
            zeta_weights.append([l_block._conv_module.output_area] *
                                l_block.num_gates)
        zeta_weights = [z for k in zeta_weights for z in k]
        return zeta_weights / np.sum(zeta_weights)

    def plot_zt(self):
        """Plots the distribution of zeta_t and returns the same."""
        zetas = self.give_zetas()
        exactly_zeros = np.sum(np.array(zetas) == 0.0)
        exactly_ones = np.sum(np.array(zetas) == 1.0)
        plt.hist(zetas)
        plt.show()
        return exactly_zeros, exactly_ones

    def get_crispnessLoss(self):
        """Loss responsible for making zeta_t 1 or 0."""
        loss = torch.FloatTensor([]).to(self.device)
        for l_block in self.prunable_modules:
            loss = torch.cat([
                loss,
                torch.pow(l_block.get_zeta_t() - l_block.get_zeta_i(), 2)
            ])
        return torch.mean(loss).to(self.device)

    def prune_model(
        self,
        target_budget,
        budget_type='channel_ratio',
        finetuning=False,
        threshold=None,
    ):
        """Prunes the network to make zeta_t exactly 1 and 0."""

        if budget_type == 'parameter_ratio':
            zetas = sorted(self.give_zetas())
            high = len(zetas) - 1
            low = 0
            while low < high:
                mid = (high + low) // 2
                threshold = zetas[mid]
                for l_block in self.prunable_modules:
                    l_block.prune(threshold)
                self.remove_orphans()
                if self.params() < target_budget:
                    high = mid - 1
                else:
                    low = mid + 1
        elif budget_type == 'flops_ratio':
            zetas = sorted(self.give_zetas())
            high = len(zetas) - 1
            low = 0
            while low < high:
                mid = (high + low) // 2
                threshold = zetas[mid]
                for l_block in self.prunable_modules:
                    l_block.prune(threshold)
                self.remove_orphans()
                if self.flops() < target_budget:
                    high = mid - 1
                else:
                    low = mid + 1
        else:
            if threshold == None:
                self.prune_threshold = self.calculate_prune_threshold(
                    target_budget, budget_type)
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
        """Unprunes the network to again make pruning gates continuous."""
        for l_block in self.prunable_modules:
            l_block.unprune()

    def prepare_for_finetuning(self, budget, budget_type='channel_ratio'):
        """Freezes zeta."""
        self.model(torch.rand(2, 3, 32, 32).to(self.device))
        threshold = self.prune_model(budget,
                                     budget_type=budget_type,
                                     finetuning=True)
        if budget_type not in ['parameter_ratio', 'flops_ratio']:
            while self.get_remaining(steepness=20.0,
                                     budget_type=budget_type) < budget:
                threshold -= 0.0001
                self.prune_model(
                    budget,
                    finetuning=True,
                    budget_type=budget_type,
                    threshold=threshold,
                )
        return threshold

    def get_params_count(self):
        """Returns the number of active and total parameters in the network."""
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
        """Returns the active and total volume of the network."""
        total_volume = 0.0
        active_volume = 0.0
        for l_block in self.prunable_modules:
            active_volume_, total_volume_ = l_block.get_volume()
            active_volume += active_volume_
            total_volume += total_volume_
        return active_volume, total_volume

    def get_flops(self):
        """Returns the active and total flops of the network."""
        total_flops = 0.0
        active_flops = 0.0
        for l_block in self.prunable_modules:
            active_flops_, total_flops_ = l_block.get_flops()
            active_flops += active_flops_
            total_flops += total_flops_
        return active_flops, total_flops

    def get_channels(self):
        """Returns the active and total number of channels in the network."""
        total_channels = 0.0
        active_channels = 0.0
        for l_block in self.prunable_modules:
            active_channels += l_block.pruned_zeta.sum().item()
            total_channels += l_block.num_gates
        return active_channels, total_channels

    def set_beta_gamma(self, beta, gamma):
        """Explicitly sets the values of beta and gamma parameters."""
        for l_block in self.prunable_modules:
            l_block.set_beta_gamma(beta, gamma)

    def check_abnormality(self):
        """Checks for any abnormality in the pruning process."""
        n_removable = self.removable_orphans()
        isbroken = self.check_if_broken()
        if n_removable != 0.0 and isbroken:
            return f'both rem_{n_removable} and broken'
        if n_removable != 0.0:
            return f'removable_{n_removable}'
        if isbroken:
            return 'broken'

    def check_if_broken(self):
        """Checks if the network is broken due to abnormal pruning."""
        for bn in self.prunable_modules:
            if bn.is_imp and bn.pruned_zeta.sum() == 0:
                return True
        return False

    def removable_orphans(self):
        """Checks for orphan nodes from the network that are not being used in
        the computation graph."""
        num_removed = 0
        bn_layers = self.model.get_bn_layers()
        for l_blocks in bn_layers:
            flag = sum([self.is_all_pruned(m) for m in l_blocks])
            if flag:
                num_removed += sum([self.n_remaining(m) for m in l_blocks])
        return num_removed

    def remove_orphans(self):
        """Removes orphan nodes from the network that are not being used in the
        computation graph."""
        num_removed = 0
        bn_layers = self.model.get_bn_layers()
        for l_blocks in bn_layers:
            flag = sum([self.is_all_pruned(m) for m in l_blocks])
            if flag:
                num_removed += sum([self.n_remaining(m) for m in l_blocks])
                for m in l_blocks:
                    m.pruned_zeta.data.copy_(torch.zeros_like(m.pruned_zeta))
        return num_removed
