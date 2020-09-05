# backend
from graphgallery.backend.gpu import set_memory_growth
from graphgallery.backend.config import (epsilon, floatx, intx, backend,
                                         set_epsilon, set_floatx, set_intx, set_backend)

from graphgallery.tensor import (sparse_adj_to_edges,
                                 sparse_tensor_to_sparse_adj,
                                 sparse_adj_to_edges,
                                 edges_to_sparse_adj,
                                 asintarr, astensor,
                                 astensors)

from graphgallery.utils.data_utils import normalize_adj, normalize_x, Bunch, sample_mask
from graphgallery.tensor.transform import normalize_adj_tensor, normalize_edge_tensor

from graphgallery.utils.shape import repeat
from graphgallery.utils.type_check import *
from graphgallery.utils.tqdm import tqdm
from graphgallery.utils.degree import degree_mixing_matrix, degree_assortativity_coefficient
from graphgallery.utils.context_manager import nullcontext
from graphgallery.utils.ego import ego_graph


# Base modules
from graphgallery import nn
from graphgallery import utils
from graphgallery import sequence
from graphgallery import data
from graphgallery import backend
from graphgallery import tensor
from graphgallery import transforms


__version__ = '0.3.0'

__all__ = ['graphgallery', 'nn', 'utils', 'sequence', 'data',
           'backend', 'tensor', 'transforms', '__version__']
