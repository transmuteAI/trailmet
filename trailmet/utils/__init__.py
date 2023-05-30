import random
import os
import numpy as np
import torch

from .benchmark import ModelBenchmark

def seed_everything(seed):
    "sets the random seed to ensure reproducibility"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True