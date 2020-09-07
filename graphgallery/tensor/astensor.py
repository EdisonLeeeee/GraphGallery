import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from graphgallery.tensor import tf_tensor, th_tensor
from graphgallery import backend

def sparse_adj_to_sparse_tensor(x):
    """Converts a Scipy sparse matrix to a TensorFlow/PyTorch SparseTensor.

    Parameters
    ----------
        x: scipy.sparse.sparse
            Matrix in Scipy sparse format.
    Returns
    -------
        S: SparseTensor
            Matrix as a sparse tensor.
    """
    if backend().kind == "T":
        return tf_tensor.sparse_adj_to_sparse_tensor(x)
    else:
        return th_tensor.sparse_adj_to_sparse_tensor(x)


def astensor(x, dtype=None):
    """Convert input matrices to Tensor or SparseTensor.

    Parameters:
    ----------
    x: tf.Tensor, tf.Variable, Scipy sparse matrix, 
        Numpy array-like, etc.

    dtype: The type of Tensor `x`, if not specified,
        it will automatically using appropriate data type.
        See `graphgallery.infer_type`.

    Returns:
    ----------      
        Tensor or SparseTensor with dtype:       
        1. `graphgallery.floatx()` if `x` is floating
        2. `graphgallery.intx() ` if `x` is integer
        3. `Bool` if `x` is bool.
    """
    if backend().kind == "T":
        return tf_tensor.astensor(x)
    else:
        return th_tensor.astensor(x)


def astensors(*xs):
    """Convert input matrices to Tensor(s) or SparseTensor(s).

    Parameters:
    ----------
    xs: tf.Tensor, tf.Variable, Scipy sparse matrix, 
        Numpy array-like, or a list of them, etc.

    Returns:
    ----------      
        Tensor(s) or SparseTensor(s) with dtype:       
        1. `graphgallery.floatx()` if `x` in `xs` is floating
        2. `graphgallery.intx() ` if `x` in `xs` is integer
        3. `Bool` if `x` in `xs` is bool.
    """
    if backend().kind == "T":
        return tf_tensor.astensors(*xs)
    else:
        return th_tensor.astensors(*xs)


def normalize_adj_tensor(adj, rate=-0.5, self_loop=1.0):
    if backend().kind == "T":
        return tf_tensor.normalize_adj_tensor(adj, rate=rate, self_loop=self_loop)
    else:
        return th_tensor.normalize_adj_tensor(adj, rate=rate, self_loop=self_loop)



def add_self_loop_edge(edge_index, edge_weight, n_nodes=None, fill_weight=1.0):
    if backend().kind == "T":
        return tf_tensor.normalize_adj_tensor(edge_index, edge_weight, n_nodes=n_nodes, fill_weight=fill_weight)
    else:
        return th_tensor.normalize_adj_tensor(edge_index, edge_weight, n_nodes=n_nodes, fill_weight=fill_weight)



def normalize_edge_tensor(edge_index, edge_weight=None, n_nodes=None, fill_weight=1.0, rate=-0.5):
    if backend().kind == "T":
        return tf_tensor.normalize_adj_tensor(edge_index, edge_weight=edge_weight, n_nodes=n_nodes, fill_weight=fill_weight, rate=rate)
    else:
        return th_tensor.normalize_adj_tensor(edge_index, edge_weight=edge_weight, n_nodes=n_nodes, fill_weight=fill_weight, rate=rate)


def sparse_tensor_to_sparse_adj(x):
    """Converts a SparseTensor to a Scipy sparse matrix (CSR matrix)."""
    if backend().kind == "T":
        return tf_tensor.sparse_tensor_to_sparse_adj(x)
    else:
        return th_tensor.sparse_tensor_to_sparse_adj(x)

