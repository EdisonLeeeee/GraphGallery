import dgl
import torch
import random
import numpy as np
from numbers import Number

__all__ = ["set_seed"]


def set_seed(seed: int):
    assert seed is None or isinstance(seed, Number), seed
    np.random.seed(seed)
    random.seed(seed)
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        dgl.random.seed(seed)
        # torch.cuda.manual_seed_all(seed)
