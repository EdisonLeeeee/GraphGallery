import torch
import numpy as np
import scipy.sparse as sp
import graphgallery as gg
from typing import Any


from graphgallery import functional as F

__all__ = ["astensor", "data_type_dict",
           "sparse_adj_to_sparse_tensor",
           "sparse_tensor_to_sparse_adj",
           "sparse_edge_to_sparse_tensor",
           "normalize_adj_tensor",
           "add_selfloops_edge",
           "normalize_edge_tensor"]

_DTYPE_TO_CLASS = {'torch.float16': "HalfTensor",
                   'torch.float32': "FloatTensor",
                   'torch.float64': "DoubleTensor",
                   'torch.int8': "CharTensor",
                   'torch.int16': "ShortTensor",
                   'torch.int32': "IntTensor",
                   'torch.int64': "LongTensor",
                   'torch.bool': "BoolTensor"}


_TYPE = {'float16': torch.float16,
         'float32': torch.float32,
         'float64': torch.float64,
         'uint8': torch.uint8,
         'int8': torch.int8,
         'int16': torch.int16,
         'int32': torch.int32,
         'int64': torch.int64,
         'bool': torch.bool}


def data_type_dict():
    return _TYPE


def dtype_to_tensor_class(dtype):
    tensor_class = _DTYPE_TO_CLASS.get(str(dtype), None)
    if tensor_class is None:
        raise ValueError(f"Invalid dtype '{dtype}'!")
    return tensor_class


def astensor(x, *, dtype=None, device=None, escape=None):

    if x is None:
        return x
    if escape is not None and isinstance(x, escape):
        return x

    if dtype is None:
        dtype = gg.infer_type(x)
    elif isinstance(dtype, str):
        ...
        # TODO
    elif isinstance(dtype, torch.dtype):
        dtype = str(dtype).split('.')[-1]
    else:
        raise TypeError(
            f"argument 'dtype' must be torch.dtype or str, not {type(dtype).__name__}.")

    if gg.is_tensor(x, backend='torch'):
        tensor = x.to(getattr(torch, dtype))
    elif gg.is_tensor(x, backend='tensorflow'):
        from ..tensor import tensoras
        return astensor(tensoras(x), dtype=dtype, device=device, escape=escape)
    elif sp.isspmatrix(x):
        if gg.backend() == "dgl_torch":
            try:
                import dgl
                tensor = dgl.from_scipy(x, idtype=getattr(torch, gg.intx()))
            except ImportError:
                tensor = sparse_adj_to_sparse_tensor(x, dtype=dtype)
        else:
            tensor = sparse_adj_to_sparse_tensor(x, dtype=dtype)

    elif isinstance(x, (np.ndarray, np.matrix)) or gg.is_listlike(x) or gg.is_scalar(x):
        tensor = torch.tensor(x, dtype=getattr(torch, dtype), device=device)
    else:
        raise TypeError(
            f'Invalid type of inputs data. Allowed data type `(Tensor, SparseTensor, Numpy array, Scipy sparse tensor, None)`, but got {type(x)}.')
    return tensor.to(device)


def sparse_edge_to_sparse_tensor(edge_index: np.ndarray, edge_weight: np.ndarray = None, shape: tuple = None) -> torch.sparse.Tensor:
    """
    edge_index: shape [2, M]
    edge_weight: shape [M,]
    """
    edge_index = F.edge_transpose(edge_index)
    edge_index = torch.LongTensor(edge_index)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], dtype=getattr(torch, gg.floatx()))
    else:
        edge_weight = torch.tensor(edge_weight)

    if shape is None:
        N = (edge_index).max() + 1
        shape = (N, N)

    shape = torch.Size(shape)
    dtype = str(edge_weight.dtype)
    return getattr(torch.sparse, dtype_to_tensor_class(dtype))(edge_index,
                                                               edge_weight,
                                                               shape)


def sparse_adj_to_sparse_tensor(x, dtype=None):
    """Converts a Scipy sparse matrix to a tensorflow SparseTensor.

    Parameters
    ----------
    x: scipy.sparse_matrix
        Matrix in Scipy sparse format.

    dtype: The type of sparse matrix `x`, if not specified,
        it will automatically using appropriate data type.
        See `graphgallery.infer_type`.

    Returns
    -------
    S: torch.sparse.FloatTensor
        Matrix as a sparse FloatTensor.
    """

    if isinstance(dtype, torch.dtype):
        dtype = str(dtype).split('.')[-1]
    elif dtype is None:
        dtype = gg.infer_type(x)

    edge_index, edge_weight = F.sparse_adj_to_edge(x)

    return sparse_edge_to_sparse_tensor(edge_index, edge_weight.astype(dtype, copy=False), x.shape)


def sparse_tensor_to_sparse_adj(x: torch.sparse.Tensor) -> sp.csr_matrix:
    """Converts a SparseTensor to a Scipy sparse matrix (CSR matrix)."""
    x = x.coalesce()
    data = x.values().detach().cpu().numpy()
    indices = x.indices().detach().cpu().numpy()
    shape = tuple(x.size())
    return sp.csr_matrix((data, indices), shape=shape)


def normalize_adj_tensor(adj, rate=-0.5, fill_weight=1.0):
    ...


def add_selfloops_edge(edge_index, edge_weight, num_nodes=None, fill_weight=1.0):

    ...


def normalize_edge_tensor(edge_index, edge_weight=None, num_nodes=None, fill_weight=1.0, rate=-0.5):

    ...
