from .torch_keras import TorchKeras
from . import tensorflow
from . import pytorch

__all__ = ['TorchKeras']

# from graphgallery import backend

# _backend = backend()
# if _backend == "pyg":
#     from .pyg import *
# elif backend == "dgl_torch":
#     from .dgl_torch import *
# elif backend == "dgl_tf":
#     from .dgl_tf import *
