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

__all__ = ['asarray', 'index_to_mask',
           'repeat', 'get_length',
           'nx_graph_to_sparse_adj', 
           'largest_indices', 'least_indices']


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


def index_to_mask(indices: np.ndarray, shape: tuple) -> np.ndarray:
    mask = np.zeros(shape, dtype=gg.boolx())
    mask[indices] = True
    return mask


def repeat(src: Any, length: int=None) -> Any:
    if src == [] or src == ():
        return []
    if length is None:
        length = get_length(src)
    if any((gg.is_scalar(src), isinstance(src, str), src is None)):
        return list(itertools.repeat(src, length))
    if len(src) > length:
        return src[:length]
    if len(src) < length:
        return list(src) + list(itertools.repeat(src[-1], length - len(src)))
    return src


def get_length(obj: Any) -> int:
    if gg.is_iterable(obj):
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


def largest_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n largest indices from a numpy array.
    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select
    Returns:
        tuple[np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    flat = array.ravel()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, array.shape)


def least_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n least indices from a numpy array.
    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select
    Returns:
        tuple[np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    flat = array.ravel()
    indices = np.argpartition(flat, n)[:n]
    indices = indices[np.argsort(flat[indices])]
    return np.unravel_index(indices, array.shape)

