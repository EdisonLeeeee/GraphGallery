import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from numbers import Number
from tensorflow.keras import backend as K
from graphgallery import config


def normalize_adj_tensor(adj, rate=-0.5, self_loop=1.0):
    adj = adj + self_loop*tf.eye(adj.shape[0], dtype=adj.dtype)
    row_sum = tf.reduce_sum(adj, axis=1)
    d_inv_sqrt = tf.pow(row_sum, rate)
    d_mat_inv_sqrt = tf.linalg.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def add_self_loop_edge(edge_index, edge_weight, n_nodes=None, fill_weight=1.0):

    if n_nodes is None:
        n_nodes = tf.reduce_max(edge_index) + 1
        
    if edge_weight is None:
        edge_weight = tf.ones([edge_index.shape[0]], dtype=config.floatx())        

    range_arr = tf.range(n_nodes, dtype=config.intx())
    diagnal_edge_index = tf.stack([range_arr, range_arr], axis=1)
    updated_edge_index = tf.concat([edge_index, diagnal_edge_index], axis=0)

    diagnal_edge_weight = tf.zeros([n_nodes], dtype=config.floatx()) + fill_weight
    updated_edge_weight = tf.concat([edge_weight, diagnal_edge_weight], axis=0)

    return updated_edge_index, updated_edge_weight


def normalize_edge_tensor(edge_index, edge_weight=None, n_nodes=None, fill_weight=1.0, rate=-0.5):

    if edge_weight is None:
        edge_weight = tf.ones([edge_index.shape[0]], dtype=config.floatx())

    if n_nodes is None:
        n_nodes = tf.reduce_max(edge_index) + 1

    edge_index, edge_weight = add_self_loop_edge(edge_index, edge_weight, n_nodes=n_nodes, fill_weight=fill_weight)

    row, col = tf.unstack(edge_index, axis=1)
    deg = tf.math.unsorted_segment_sum(edge_weight, row, num_segments=n_nodes)
    deg_inv_sqrt = tf.pow(deg, rate)

    # check if exists NAN
    deg_inv_sqrt = tf.where(
        tf.math.logical_or(tf.math.is_inf(deg_inv_sqrt), tf.math.is_nan(deg_inv_sqrt)),
        tf.zeros_like(deg_inv_sqrt),
        deg_inv_sqrt
    )

    edge_weight_norm = tf.gather(deg_inv_sqrt, row) * edge_weight * tf.gather(deg_inv_sqrt, col)

    return edge_index, edge_weight_norm
