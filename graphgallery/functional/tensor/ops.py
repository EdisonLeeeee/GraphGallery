import torch
import numpy as np
from typing import Any, Union, Optional

import scipy.sparse as sp
import graphgallery as gg
from graphgallery import functional as gf

__all__ = ["sparse_adj_to_sparse_tensor",
           "sparse_tensor_to_sparse_adj",
           "sparse_edge_to_sparse_tensor",
           "infer_type",
           ]


_floatx = 'float32'
_intx = 'int64'


def infer_type(x: Any) -> str:
    f"""Infer the type of the input 'x'.

    Parameters:
    ----------
    x: Any python object

    Returns:
    ----------
    dtype: string, the proper data type of 'x':
        1. '{_floatx}' if 'x' is floating,
        2. '{_intx}' if 'x' is integer,
    """
    # For tensor or variable
    if torch.is_tensor(x):
        if x.dtype.is_floating_point:
            return _floatx
        elif x.dtype == torch.bool:
            return 'bool'
        elif 'int' in str(x.dtype):
            return _intx
        else:
            raise TypeError(f"Invalid type of pytorch Tensor: '{type(x)}'")
    _x = x
    if not hasattr(_x, 'dtype'):
        _x = np.asarray(_x)

    if _x.dtype.kind in {'f', 'c'}:
        return _floatx
    elif _x.dtype.kind in {'i', 'u'}:
        return _intx
    elif _x.dtype.kind == 'b':
        return 'bool'
    elif _x.dtype.kind == 'O':
        raise TypeError(f"Invalid inputs of '{x}'.")
    else:
        raise TypeError(f"Invalid input of '{type(x).__name__}'.")


def sparse_adj_to_sparse_tensor(x, *, dtype=None):
    """Converts a Scipy sparse matrix to a PyTorch SparseTensor.

    Parameters
    ----------
    x: Scipy sparse matrix
        Matrix in Scipy sparse format.

    Returns
    -------
    S: SparseTensor
        Matrix as a pytorch sparse tensor.
    """

    if dtype is None:
        dtype = infer_type(x)

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
