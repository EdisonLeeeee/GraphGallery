import torch
import numpy as np
import scipy.sparse as sp

from graphgallery import floatx, intx
from graphgallery.utils.type_check import (is_list_like,
                                           is_interger_scalar,
                                           is_tensor,
                                           is_scalar_like,
                                           infer_type)


from graphgallery.utils.decorators import MultiInputs
from graphgallery import transforms as T

__all__ = ["astensor", "astensors",
           "sparse_adj_to_sparse_tensor",
           "sparse_tensor_to_sparse_adj",
           "sparse_edges_to_sparse_tensor",
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


def dtype_to_tensor_class(dtype):
    tensor_class = _DTYPE_TO_CLASS.get(str(dtype), None)
    if tensor_class is None:
        raise RuntimeError(f"Invalid dtype '{dtype}'!")
    return tensor_class


def astensor(x, dtype=None, device=None):
    """Convert input matrices to Tensor or SparseTensor.

    Parameters:
    ----------
    x: any python object.

    dtype: The type of Tensor `x`, if not specified,
        it will automatically use appropriate data type.
        See `graphgallery.infer_type`.

    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.

    Returns:
    ----------      
    Tensor(s) or SparseTensor(s) with dtype, if dtype is `None`:        
        1. `graphgallery.floatx()` if `x` is floating
        2. `graphgallery.intx() ` if `x` is integer
        3. `'bool'` if `x` is bool.
    """
    if x is None:
        return x

    if dtype is None:
        dtype = infer_type(x)
    elif isinstance(dtype, str):
        ...
        # TODO
    elif isinstance(dtype, torch.dtype):
        dtype = str(dtype).split('.')[-1]
    else:
        raise TypeError(
            f"argument 'dtype' must be torch.dtype or str, not {type(dtype).__name__}.")

    if is_tensor(x, "P"):
        tensor = x.to(getattr(torch, dtype))
    elif sp.isspmatrix(x):
        tensor = sparse_adj_to_sparse_tensor(x, dtype=dtype)
    elif isinstance(x, (np.ndarray, np.matrix)) or is_list_like(x) or is_scalar_like(x):
        tensor = torch.tensor(x, dtype=getattr(torch, dtype), device=device)
    else:
        raise TypeError(
            f'Invalid type of inputs data. Allowed data type `(Tensor, SparseTensor, Numpy array, Scipy sparse tensor, None)`, but got {type(x)}.')

    return tensor.to(device)


astensors = MultiInputs(type_check=False)(astensor)
astensors.__doc__ = """Convert input matrices to Tensor(s) or SparseTensor(s).

    Parameters:
    ----------
    xs: one or a list python object(s)

    dtype: The type of Tensor for all objects in `xs`, if not specified,
        it will automatically use appropriate data type.
        See `graphgallery.infer_type`.
        
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
        
    Returns:
    ----------      
    Tensor(s) or SparseTensor(s) with dtype, if dtype is `None`:    
        1. `graphgallery.floatx()` if `x` in `xs` is floating
        2. `graphgallery.intx() ` if `x` in `xs` is integer
        3. `'bool'` if `x` in 'xs' is bool.
    """


def sparse_edges_to_sparse_tensor(edge_index: np.ndarray, edge_weight: np.ndarray = None, shape: tuple = None) -> torch.sparse.Tensor:
    """
    edge_index: shape [2, M]
    edge_weight: shape [M,]
    """
    edge_index = T.edge_transpose(edge_index)
    edge_index = torch.LongTensor(edge_index)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], dtype=getattr(torch, floatx()))
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
    x: scipy.sparse.sparse
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
        dtype = infer_type(x)

    edge_index, edge_weight = T.sparse_adj_to_sparse_edges(x)

    return sparse_edges_to_sparse_tensor(edge_index, edge_weight.astype(dtype, copy=False), x.shape)


def sparse_tensor_to_sparse_adj(x: torch.sparse.Tensor) -> sp.csr_matrix:
    """Converts a SparseTensor to a Scipy sparse matrix (CSR matrix)."""
    x = x.coalesce()
    data = x.values().detach().cpu().numpy()
    indices = x.indices().detach().cpu().numpy()
    shape = tuple(x.size())
    return sp.csr_matrix((data, indices), shape=shape)


def normalize_adj_tensor(adj, rate=-0.5, fill_weight=1.0):
    ...


def add_selfloops_edge(edge_index, edge_weight, n_nodes=None, fill_weight=1.0):

    ...


def normalize_edge_tensor(edge_index, edge_weight=None, n_nodes=None, fill_weight=1.0, rate=-0.5):

    ...
