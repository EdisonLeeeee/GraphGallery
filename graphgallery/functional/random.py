import random
import torch
import numpy as np
import graphgallery as gg

__all__ = ["random_seed"]


def random_seed(seed=None, backend=None):
    backend = gg.backend(backend)
    np.random.seed(seed)
    random.seed(seed)
    if backend == "torch":
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
