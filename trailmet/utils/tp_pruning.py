import torch
import os
import yaml
import torch_pruning as tp
import numpy as np
from .benchmark import ModelBenchmark
from ..algorithms.prune.network_slimming import Network_Slimming
from ..algorithms.prune.pns import SlimPruner, ChannelRounding
from ..algorithms.prune.functional import cal_threshold_by_bn2d_weights

class TP_Prune:
    def __init__(self, method=None, prune_model=None, org_model=None, batch_size=64, input_size=32, device_name="cpu", root=None, schema=None):
        self.root = root
        self.method = method
        self.prune_model = prune_model
        self.org_model = org_model
        self.batch_size = batch_size
        self.input_size = input_size
        self.device = torch.device(device_name)
        self.prune_model.to(self.device)
        self.org_model.to(self.device)

        # for network slimming
        if self.method == "network_slimming":
            with open(os.path.join(self.root, "resnet50_cifar100.yaml"), 'r') as stream:
                data_loaded = yaml.safe_load(stream)
            data_loaded['schema_root'] = self.root
            slim = Network_Slimming(**data_loaded)
            self.pruner = SlimPruner(self.prune_model,slim.prune_schema)
            self.threshold = cal_threshold_by_bn2d_weights(
                        [it.module for it in self.pruner.bn2d_modules.values()], slim.prune_ratio
                    )
    def prune(self):
        
        DG = tp.DependencyGraph().build_dependency(self.org_model, example_inputs=torch.randn(1,3,self.input_size, self.input_size).to(self.device))

        if self.method == "chipnet":
            print("==> Chipnet method is selected for pruning")
            print("==> Pruning model started")
            
            for prune_model_modules, org_model_modules in zip(self.prune_model.named_modules(), self.org_model.named_modules()):
                if hasattr(prune_model_modules[1], "pruned_zeta") and (0 in prune_model_modules[1].pruned_zeta):

                    indices = torch.where(prune_model_modules[1].pruned_zeta == 0)[0].cpu().tolist()
                    group = DG.get_pruning_group( org_model_modules[1], tp.prune_batchnorm_out_channels, idxs=indices )

                    # # 3. prune all grouped layers that are coupled with model.conv1 (included).
                    if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.
                        group.prune()
            
            print("==> Pruning model completed.")

        elif self.method == "network_slimming":
            print("==> Network Slimming method is selected for pruning")
            print("==> Pruning model started")

            for name, module in self.org_model.named_modules():
                if "model." + name in list(self.pruner.bn2d_modules.keys()):
                    self.pruner.bn2d_modules["model." + name].cal_keep_idxes(self.threshold,
                    min_keep_ratio=0.02,
                    channel_rounding=ChannelRounding(self.pruner.channel_rounding),
                )

                    total_idxs = np.arange(self.pruner.bn2d_modules["model." + name].module.num_features)
                    remove_idxs = np.setdiff1d(total_idxs, self.pruner.bn2d_modules["model." + name].keep_idxes)

            
                    group = DG.get_pruning_group( module, tp.prune_batchnorm_out_channels, idxs=remove_idxs.tolist() )

                    # # 3. prune all grouped layers that are coupled with model.conv1 (included).
                    if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.
                        group.prune()

            
        return self.prune_model, self.org_model

    def benchmark_model(self, model):
        print("==> Model for benchmarking")
        print(model)
        print("==> Benchmarking model started")
        model_benchmark = ModelBenchmark(model, self.batch_size, self.input_size, self.device)
        model_benchmark.benchmark()
        