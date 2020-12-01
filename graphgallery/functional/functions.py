import torch
import itertools

import numpy as np
import networkx as nx
import tensorflow as tf
import scipy.sparse as sp

from typing import Any, Optional
from numbers import Number

import graphgallery as gg
from graphgallery import functional as gf

__all__ = ['asarray', 'indices2mask',
           'repeat', 'get_length', 'onehot',
           'nx_graph_to_sparse_adj']


def asarray(x: Any, dtype: Optional[str] = None) -> np.ndarray:
    """Convert `x` to interger Numpy array.

    Parameters:
    ----------
    x: Tensor, Scipy sparse matrix,
        Numpy array-like, etc.

    Returns:
    ----------
    Integer Numpy array with dtype or `graphgallery.intx()`

    """
    if dtype is None:
        dtype = gg.intx()

    if gf.is_tensor(x, backend="tensorflow"):
        if x.dtype != dtype:
            return tf.cast(x, dtype=dtype)
        else:
            return x

    if gf.is_tensor(x, backend="torch"):
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


def indices2mask(indices: np.ndarray, shape: tuple) -> np.ndarray:
    mask = np.zeros(shape, dtype=gg.boolx())
    mask[indices] = True
    return mask


def repeat(src: Any, length: int) -> Any:
    if src is None:
        return [None for _ in range(length)]
    if src == [] or src == ():
        return []
    if isinstance(src, (Number, str)):
        return list(itertools.repeat(src, length))
    if (len(src) > length):
        return src[:length]
    if (len(src) < length):
        return list(src) + list(itertools.repeat(src[-1], length - len(src)))
    return src


def get_length(obj: Any) -> int:
    if gg.is_iterable(obj):
        length = len(obj)
    else:
        length = 1
    return length


def onehot(label):
    """Get the one-hot like label of nodes."""
    label = np.asarray(label, dtype=np.int32)
    if label.ndim == 1:
        return np.eye(label.max() + 1, dtype=label.dtype)[label]
    else:
        raise ValueError(f"label must be a 1D array, but got {label.ndim}D array.")


def nx_graph_to_sparse_adj(graph):
    num_nodes = graph.number_of_nodes()
    data = np.asarray(list(graph.edges().data('weight', default=1.0)))
    edge_index = data[:, :2].T.astype(np.int64)
    edge_weight = data[:, -1].T.astype(np.float32)
    adj_matrix = sp.csr_matrix((edge_weight, edge_index), shape=(num_nodes, num_nodes))
    return adj_matrix


