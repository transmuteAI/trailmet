from .l1_pruner import Pruner as L1Pruner
from .meta_pruner import MetaPruner
from .reg_pruner import Pruner as RegPruner

# when new pruner implementation is added in the 'pruner' dir, update this dict to maintain minimal code change.
# key: pruning method name, value: the corresponding pruner
pruner_dict = {
    'GReg-1': RegPruner,
    'GReg-2': RegPruner,
    'L1': L1Pruner,
}
