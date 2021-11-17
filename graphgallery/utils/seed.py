import dgl
import torch
import random
import numpy as np
from numbers import Number
from typing import Optional
from graphgallery import backend

__all__ = ["set_seed"]


def set_seed(seed: Optional[int] = None):
    assert seed is None or isinstance(seed, Number), seed
    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if backend() == 'dgl':
            dgl.random.seed(seed)
        # torch.cuda.manual_seed_all(seed)
