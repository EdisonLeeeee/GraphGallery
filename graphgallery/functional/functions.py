import torch
import itertools

import numpy as np
import scipy.sparse as sp
from numbers import Number
from typing import Any, Optional

import graphgallery as gg

__all__ = ['asarray', 'index_to_mask',
           'repeat', 'get_length',
           'nx_graph_to_sparse_adj']


def asarray(x: Any, dtype: Optional[str] = None) -> np.ndarray:
    """Convert `x` to interger Numpy array.

    Parameters:
    ----------
    x: Tensor, Scipy sparse matrix,
        Numpy array-like, etc.

    Returns:
    ----------
    Integer Numpy array with dtype or `'int64'`

    """
    if dtype is None:
        dtype = 'int64'

    if torch.is_tensor(x):
        if x.dtype != dtype:
            return x.to(getattr(torch, dtype))
        else:
            return x

    if gg.is_intscalar(x):
        x = np.asarray([x], dtype=dtype)
    elif gg.is_listlike(x) or (isinstance(x, np.ndarray) and x.dtype != "O"):
        x = np.asarray(x, dtype=dtype)
    else:
        raise ValueError(
            f"Invalid input which should be either array-like or integer scalar, but got {type(x)}.")
    return x


def index_to_mask(indices: np.ndarray, shape: tuple) -> np.ndarray:
    mask = np.zeros(shape, dtype=gg.boolx())
    mask[indices] = True
    return mask


def repeat(src: Any, length: Optional[int] = None) -> Any:
    """repeat any objects and return iterable ones.

    Parameters
    ----------
    src : Any
        any objects
    length : Optional[int], optional
        the length to be repeated. If `None`,
        it would return the iterable object itself, by default None

    Returns
    -------
    Any
        the iterable repeated object


    Example
    -------
    >>> from graphwar.utils import repeat
    # repeat for single non-iterable object
    >>> repeat(1)
    [1]
    >>> repeat(1, 3)
    [1, 1, 1]
    >>> repeat('relu', 2)
    ['relu', 'relu']
    >>> repeat(None, 2)
    [None, None]
    # repeat for iterable object
    >>> repeat([1, 2, 3], 2)
    [1, 2]
    >>> repeat([1, 2, 3], 5)
    [1, 2, 3, 3, 3]

    """
    if src == [] or src == ():
        return []
    if length is None:
        length = get_length(src)
    if any((isinstance(src, Number), isinstance(src, str), src is None)):
        return list(itertools.repeat(src, length))
    if len(src) > length:
        return src[:length]
    if len(src) < length:
        return list(src) + list(itertools.repeat(src[-1], length - len(src)))
    return src


def get_length(obj: Any) -> int:
    if isinstance(obj, (list, tuple)):
        length = len(obj)
    else:
        length = 1
    return length


def nx_graph_to_sparse_adj(graph):
    num_nodes = graph.number_of_nodes()
    data = np.asarray(list(graph.edges().data('weight', default=1.0)))
    edge_index = data[:, :2].T.astype(np.int64)
    edge_weight = data[:, -1].T.astype(np.float32)
    adj_matrix = sp.csr_matrix((edge_weight, edge_index), shape=(num_nodes, num_nodes))
    return adj_matrix
