import torch
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from graphgallery import floatx, intx
from graphgallery.utils.type_check import (is_list_like,
                                           is_interger_scalar,
                                           is_tensor_or_variable,
                                           is_scalar_like)


from graphgallery.utils.decorators import MultiInputs
from graphgallery import transforms as T


_DTYPE_TO_CLASS = {'float16': "HalfTensor",
                   'float32': "FloatTensor",
                   'float64': "DoubleTensor",
                   'int8': "CharTensor",
                   'int16': "ShortTensor",
                   'int32': "IntTensor",
                   'int64': "LongTensor",
                   'bool': "BoolTensor"}


def dtype_to_tensor_class(dtype):
    tensor_class = _DTYPE_TO_CLASS.get(str(dtype), None)
    if tensor_class is None:
        raise RuntimeError(f"Invalid dtype '{dtype}'!")
    return tensor_class


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
    edge_index = torch.LongTensor(edge_index)
    edge_weight = torch.tensor(edge_weight, dtype=getattr(torch, dtype))
    return getattr(torch.sparse, dtype_to_tensor_class(dtype))(edge_index,
                                                               edge_weight,
                                                               torch.Size(x.shape))


def infer_type(x):
    """Infer type of the input `x`.

     Parameters:
    ----------
    x: tf.Tensor, tf.Variable, Scipy sparse matrix,
        Numpy array-like, etc.

    Returns:
    ----------
    dtype: string, the converted type of `x`:
        1. `graphgallery.floatx()` if `x` is floating
        2. `graphgallery.intx()` if `x` is integer
        3. `'bool'` if `x` is bool.

    """
    # For tensor or variable
    if is_tensor_or_variable(x):
        if x.dtype.is_floating_point:
            return floatx()
        elif x.dtype == torch.bool:
            return 'bool'
        elif 'int' in str(x.dtype):
            return intx()
        else:
            raise RuntimeError(f'Invalid input of `{type(x)}`')

    if not hasattr(x, 'dtype'):
        x = np.asarray(x)

    if x.dtype.kind in {'f', 'c'}:
        return floatx()
    elif x.dtype.kind in {'i', 'u'}:
        return intx()
    elif x.dtype.kind == 'b':
        return 'bool'
    elif x.dtype.kind == 'O':
        raise RuntimeError(f'Invalid inputs of `{x}`.')
    else:
        raise RuntimeError(f'Invalid input of `{type(x)}`')


def astensor(x, dtype=None, device=None):
    """Convert input matrices to Tensor or SparseTensor.

    Parameters:
    ----------
    x: tf.Tensor, tf.Variable, Scipy sparse matrix, 
        Numpy array-like, etc.

    dtype: The type of Tensor `x`, if not specified,
        it will automatically using appropriate data type.
        See `graphgallery.infer_type`.

    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.

    Returns:
    ----------      
    Tensor or SparseTensor with dtype:       
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

    if is_tensor_or_variable(x):
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
    xs: tf.Tensor, tf.Variable, Scipy sparse matrix, 
        Numpy array-like, or a list of them, etc.

    dtype: The type of Tensor for all tensors in `xs`, if not specified,
        it will automatically using appropriate data type.
        See `graphgallery.infer_type`.
        
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
        
    Returns:
    ----------      
    Tensor(s) or SparseTensor(s) with dtype:       
        1. `graphgallery.floatx()` if `x` in `xs` is floating
        2. `graphgallery.intx() ` if `x` in `xs` is integer
        3. `'bool'` if `x` in 'xs' is bool.
    """


def normalize_adj_tensor(adj, rate=-0.5, fill_weight=1.0):
    ...


def add_selfloops_edge(edge_index, edge_weight, n_nodes=None, fill_weight=1.0):

    ...


def normalize_edge_tensor(edge_index, edge_weight=None, n_nodes=None, fill_weight=1.0, rate=-0.5):

    ...


def sparse_tensor_to_sparse_adj(x):
    """Converts a SparseTensor to a Scipy sparse matrix (CSR matrix)."""
    ...
