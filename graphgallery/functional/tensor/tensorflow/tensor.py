import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import graphgallery as gg

from graphgallery import functional as F
from typing import Any


__all__ = ["astensor", "data_type_dict",
           "sparse_adj_to_sparse_tensor",
           "sparse_tensor_to_sparse_adj",
           "sparse_edge_to_sparse_tensor",
           "normalize_adj_tensor",
           "add_selfloops_edge",
           "normalize_edge_tensor"]

_TYPE = {'float16': tf.float16,
         'float32': tf.float32,
         'float64': tf.float64,
         'uint8': tf.uint8,
         'int8': tf.int8,
         'int16': tf.int16,
         'int32': tf.int32,
         'int64': tf.int64,
         'bool': tf.bool}


def data_type_dict():
    """
    Return a dict type of a dict.

    Args:
    """
    return _TYPE


def astensor(x, *, dtype=None, device=None, escape=None):
    """
    Convert x to a tensor.

    Args:
        x: (todo): write your description
        dtype: (str): write your description
        device: (todo): write your description
        escape: (todo): write your description
    """

    if x is None:
        return x
    
    if escape is not None and isinstance(x, escape):
        return x
    
    if dtype is None:
        dtype = gg.infer_type(x)
    elif isinstance(dtype, str):
        ...
        # TODO
    elif isinstance(dtype, tf.dtypes.DType):
        dtype = dtype.name
    else:
        raise TypeError(
            f"argument 'dtype' must be tensorflow.dtypes.DType or str, not {type(dtype).__name__}.")

    with tf.device(device):
        if gg.is_tensor(x, backend='tensorflow'):
            if x.dtype != dtype:
                x = tf.cast(x, dtype=dtype)
            return x
        elif gg.is_tensor(x, backend='torch'):
            from ..tensor import tensoras
            return astensor(tensoras(x), dtype=dtype, device=device, escape=escape)
        elif sp.isspmatrix(x):
            return sparse_adj_to_sparse_tensor(x, dtype=dtype)
        elif isinstance(x, (np.ndarray, np.matrix)) or gg.is_listlike(x) or gg.is_scalar(x):
            return tf.convert_to_tensor(x, dtype=dtype)
        else:
            raise TypeError(
                f'Invalid type of inputs data. Allowed data type (Tensor, SparseTensor, Numpy array, Scipy sparse matrix, None), but got {type(x)}.')


def sparse_edge_to_sparse_tensor(edge_index: np.ndarray, edge_weight: np.ndarray = None, shape: tuple = None) -> tf.SparseTensor:
    """
    edge_index: shape [2, M]
    edge_weight: shape [M,]
    """
    edge_index = F.edge_transpose(edge_index)

    if edge_weight is None:
        edge_weight = tf.ones(edge_index.shape[1], dtype=gg.floatx())

    if shape is None:
        N = (edge_index).max + 1
        shape = (N, N)

    return tf.SparseTensor(edge_index.T, edge_weight, shape)


def sparse_adj_to_sparse_tensor(x: sp.csr_matrix, dtype=None):
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
    S: tf.sparse.SparseTensor
        Matrix as a sparse tensor.
    """

    if isinstance(dtype, tf.dtypes.DType):
        dtype = dtype.name
    elif dtype is None:
        dtype = gg.gg.infer_type(x)

    edge_index, edge_weight = F.sparse_adj_to_edge(x)
    return sparse_edge_to_sparse_tensor(edge_index, edge_weight.astype(dtype, copy=False), x.shape)


def sparse_tensor_to_sparse_adj(x) -> sp.csr_matrix:
    """Converts a SparseTensor to a Scipy sparse matrix (CSR matrix)."""
    data = x.values.numpy()
    indices = x.indices.numpy().T
    shape = x.shape
    return sp.csr_matrix((data, indices), shape=shape)


def normalize_adj_tensor(adj, rate=-0.5, fill_weight=1.0):
    """
    Normalize the adjacency matrix.

    Args:
        adj: (todo): write your description
        rate: (array): write your description
        fill_weight: (str): write your description
    """
    if fill_weight:
        adj = adj + fill_weight * tf.eye(tf.shape(adj)[0], dtype=adj.dtype)
    d = tf.reduce_sum(adj, axis=1)
    d_power = tf.pow(d, rate)
    d_power_mat = tf.linalg.diag(d_power)
    return d_power_mat @ adj @ d_power_mat


def add_selfloops_edge(edge_index, edge_weight, n_nodes=None, fill_weight=1.0):
    """
    Adds an edge to edge_index.

    Args:
        edge_index: (int): write your description
        edge_weight: (todo): write your description
        n_nodes: (int): write your description
        fill_weight: (todo): write your description
    """

    if n_nodes is None:
        n_nodes = tf.reduce_max(edge_index) + 1

    if edge_weight is None:
        edge_weight = tf.ones([edge_index.shape[0]], dtype=gg.floatx())

    range_arr = tf.range(n_nodes, dtype=edge_index.dtype)
    diagnal_edge_index = tf.stack([range_arr, range_arr], axis=1)
    updated_edge_index = tf.concat([edge_index, diagnal_edge_index], axis=0)

    diagnal_edge_weight = tf.zeros([n_nodes], dtype=gg.floatx()) + fill_weight
    updated_edge_weight = tf.concat([edge_weight, diagnal_edge_weight], axis=0)

    return updated_edge_index, updated_edge_weight


def normalize_edge_tensor(edge_index, edge_weight=None, n_nodes=None, fill_weight=1.0, rate=-0.5):
    """
    Normalize the tensor.

    Args:
        edge_index: (int): write your description
        edge_weight: (todo): write your description
        n_nodes: (int): write your description
        fill_weight: (todo): write your description
        rate: (todo): write your description
    """

    if edge_weight is None:
        edge_weight = tf.ones([edge_index.shape[0]], dtype=gg.floatx())

    if n_nodes is None:
        n_nodes = tf.reduce_max(edge_index) + 1

    edge_index, edge_weight = add_selfloops_edge(
        edge_index, edge_weight, n_nodes=n_nodes, fill_weight=fill_weight)

    row, col = tf.unstack(edge_index, axis=1)
    deg = tf.math.unsorted_segment_sum(edge_weight, row, num_segments=n_nodes)
    deg_inv_sqrt = tf.pow(deg, rate)

    # check if exists NAN
    deg_inv_sqrt = tf.where(
        tf.math.logical_or(tf.math.is_inf(deg_inv_sqrt),
                           tf.math.is_nan(deg_inv_sqrt)),
        tf.zeros_like(deg_inv_sqrt),
        deg_inv_sqrt
    )

    edge_weight_norm = tf.gather(
        deg_inv_sqrt, row) * edge_weight * tf.gather(deg_inv_sqrt, col)

    return edge_index, edge_weight_norm
