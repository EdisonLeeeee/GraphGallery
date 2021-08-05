import torch
import numpy as np
import scipy.sparse as sp
import graphgallery as gg
from typing import Any

from graphgallery import functional as gf

__all__ = ["gather", "sparse_adj_to_sparse_tensor", "sparse_tensor_to_sparse_adj",
           "sparse_edge_to_sparse_tensor", "normalize_adj_tensor",
           "add_selfloops_edge_tensor", "normalize_edge_tensor"
           ]

_DTYPE_TO_CLASS = {
    'torch.float16': "HalfTensor",
    'torch.float32': "FloatTensor",
    'torch.float64': "DoubleTensor",
    'torch.int8': "CharTensor",
    'torch.int16': "ShortTensor",
    'torch.int32': "IntTensor",
    'torch.int64': "LongTensor",
    'torch.bool': "BoolTensor"
}


def gather(out, out_index):
    if out_index is not None:
        return out[out_index]
    return out


def dtype_to_tensor_class(dtype: str):
    tensor_class = _DTYPE_TO_CLASS.get(str(dtype), None)
    if not tensor_class:
        raise ValueError(f"Invalid dtype '{dtype}'!")
    return tensor_class


def sparse_edge_to_sparse_tensor(edge_index: np.ndarray,
                                 edge_weight: np.ndarray = None,
                                 shape: tuple = None) -> torch.Tensor:
    """
    edge_index: shape [2, M]
    edge_weight: shape [M,]
    """
    edge_index = gf.asedge(edge_index, shape="col_wise")
    edge_index = torch.LongTensor(edge_index)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1],
                                 dtype=getattr(torch, gg.floatx()))
    else:
        edge_weight = torch.tensor(edge_weight)

    if shape is None:
        shape = gf.maybe_shape(edge_index)

    shape = torch.Size(shape)
    dtype = str(edge_weight.dtype)
    return getattr(torch.sparse,
                   dtype_to_tensor_class(dtype))(edge_index, edge_weight,
                                                 shape)


def sparse_adj_to_sparse_tensor(x, dtype=None):

    if dtype is None:
        dtype = gf.infer_type(x)
    elif isinstance(dtype, torch.dtype):
        dtype = str(dtype).split('.')[-1]

    if not isinstance(dtype, str):
        raise TypeError(dtype)

    edge_index, edge_weight = gf.sparse_adj_to_edge(x)
    edge_weight = edge_weight.astype(dtype, copy=False)
    return sparse_edge_to_sparse_tensor(edge_index,
                                        edge_weight,
                                        x.shape)


def sparse_tensor_to_sparse_adj(x: torch.Tensor) -> sp.csr_matrix:
    """Converts a SparseTensor to a Scipy sparse matrix (CSR matrix)."""
    x = x.coalesce()
    data = x.values().detach().cpu().numpy()
    indices = x.indices().detach().cpu().numpy()
    shape = tuple(x.size())
    return sp.csr_matrix((data, indices), shape=shape)


def normalize_adj_tensor(adj, rate=-0.5, fill_weight=1.0):
    device = torch.device("cuda" if adj.is_cuda else "cpu")
    adj = adj + fill_weight * torch.eye(adj.shape[0]).to(device)
    d = adj.sum(1)
    d_power = d.pow(rate).flatten()
    d_power_mat = torch.diag(d_power)
    return d_power_mat @ adj @ d_power_mat


def add_selfloops_edge_tensor(edge_index,
                       edge_weight,
                       num_nodes=None,
                       fill_weight=1.0):
    # TODO
    ...


def normalize_edge_tensor(edge_index,
                          edge_weight=None,
                          num_nodes=None,
                          fill_weight=1.0,
                          rate=-0.5):
    # TODO
    ...
