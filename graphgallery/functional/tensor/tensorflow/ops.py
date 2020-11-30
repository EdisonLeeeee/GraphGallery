import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import graphgallery as gg

from graphgallery import functional as gf
from typing import Any


def sparse_edge_to_sparse_tensor(edge_index: np.ndarray,
                                 edge_weight: np.ndarray = None,
                                 shape: tuple = None) -> tf.SparseTensor:
    """
    edge_index: shape [2, M]
    edge_weight: shape [M,]
    """
    edge_index = gf.edge_transpose(edge_index)

    if edge_weight is None:
        edge_weight = tf.ones(edge_index.shape[1], dtype=gg.floatx())

    if shape is None:
        shape = gf.maybe_shape(edge_index)

    return tf.SparseTensor(edge_index.T, edge_weight, shape)


def sparse_adj_to_sparse_tensor(x: sp.csr_matrix, dtype=None):

    if dtype is None:
        dtype = gf.infer_type(x)
    elif isinstance(dtype, tf.dtypes.DType):
        dtype = dtype.name
    if not isinstance(dtype, str):
        raise TypeError(dtype)

    edge_index, edge_weight = gf.sparse_adj_to_edge(x)
    edge_weight = edge_weight.astype(dtype, copy=False)
    return sparse_edge_to_sparse_tensor(edge_index,
                                        edge_weight,
                                        x.shape)


def sparse_tensor_to_sparse_adj(x) -> sp.csr_matrix:
    """Converts a SparseTensor to a Scipy sparse matrix (CSR matrix)."""
    data = x.values.numpy()
    indices = x.indices.numpy().T
    shape = x.shape
    return sp.csr_matrix((data, indices), shape=shape)


def normalize_adj_tensor(adj, rate=-0.5, fill_weight=1.0):
    if fill_weight:
        adj = adj + fill_weight * tf.eye(tf.shape(adj)[0], dtype=adj.dtype)
    d = tf.reduce_sum(adj, axis=1)
    d_power = tf.pow(d, rate)
    d_power_mat = tf.linalg.diag(d_power)
    return d_power_mat @ adj @ d_power_mat


def add_selfloops_edge(edge_index,
                       edge_weight,
                       num_nodes=None,
                       fill_weight=1.0):

    if num_nodes is None:
        num_nodes = tf.reduce_max(edge_index) + 1

    if edge_weight is None:
        edge_weight = tf.ones([edge_index.shape[0]], dtype=gg.floatx())

    range_arr = tf.range(num_nodes, dtype=edge_index.dtype)
    diagnal_edge_index = tf.stack([range_arr, range_arr], axis=1)
    updated_edge_index = tf.concat([edge_index, diagnal_edge_index], axis=0)

    diagnal_edge_weight = tf.zeros([num_nodes],
                                   dtype=gg.floatx()) + fill_weight
    updated_edge_weight = tf.concat([edge_weight, diagnal_edge_weight], axis=0)

    return updated_edge_index, updated_edge_weight


def normalize_edge_tensor(edge_index,
                          edge_weight=None,
                          num_nodes=None,
                          fill_weight=1.0,
                          rate=-0.5):

    if edge_weight is None:
        edge_weight = tf.ones([edge_index.shape[0]], dtype=gg.floatx())

    if num_nodes is None:
        num_nodes = tf.reduce_max(edge_index) + 1

    edge_index, edge_weight = add_selfloops_edge(edge_index,
                                                 edge_weight,
                                                 num_nodes=num_nodes,
                                                 fill_weight=fill_weight)

    row, col = tf.unstack(edge_index, axis=1)
    deg = tf.math.unsorted_segment_sum(edge_weight,
                                       row,
                                       num_segments=num_nodes)
    deg_inv_sqrt = tf.pow(deg, rate)

    # check if exists NAN
    deg_inv_sqrt = tf.where(
        tf.math.logical_or(tf.math.is_inf(deg_inv_sqrt),
                           tf.math.is_nan(deg_inv_sqrt)),
        tf.zeros_like(deg_inv_sqrt), deg_inv_sqrt)

    edge_weight_norm = tf.gather(deg_inv_sqrt, row) * edge_weight * tf.gather(
        deg_inv_sqrt, col)

    return edge_index, edge_weight_norm
