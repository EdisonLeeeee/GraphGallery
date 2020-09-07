import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from graphgallery import floatx, intx
from graphgallery.utils.type_check import (is_list_like,
                                           is_interger_scalar,
                                           is_tensor_or_variable,
                                           is_scalar_like)


# def sparse_adj_to_sparse_tensor(x):
#     """Converts a Scipy sparse matrix to a tensorflow SparseTensor."""
#     sparse_coo = x.tocoo()
#     row, col = sparse_coo.row, sparse_coo.col
#     data, shape = sparse_coo.data, sparse_coo.shape
#     indices = np.concatenate(
#         (np.expand_dims(row, axis=1), np.expand_dims(col, axis=1)), axis=1)
#     return tf.sparse.SparseTensor(indices, data, shape)


def sparse_adj_to_sparse_tensor(x: sp.csr_matrix):
    """Converts a Scipy sparse matrix to a tensorflow SparseTensor.

    Parameters
    ----------
        x: scipy.sparse.sparse
            Matrix in Scipy sparse format.
    Returns
    -------
        S: tf.sparse.SparseTensor
            Matrix as a sparse tensor.
    """
    x = x.tocoo(copy=False)
    return tf.SparseTensor(np.vstack((x.row, x.col)).T, x.data, x.shape)


def infer_type(x):
    """Infer type of the input `x`.

     Parameters:
    ----------
    x: tf.Tensor, tf.Variable, Scipy sparse matrix,
        Numpy array-like, etc.

    Returns:
    ----------
        The converted type of `x`:
        1. `graphgallery.floatx()` if `x` is floating
        2. `graphgallery.intx() ` if `x` is integer
        3. `Bool` if `x` is bool.

    """

    # For tensor or variable
    if is_tensor_or_variable(x):
        if x.dtype.is_floating:
            return floatx()
        elif x.dtype.is_integer or x.dtype.is_unsigned:
            return intx()
        elif x.dtype.is_bool:
            return 'bool'
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

    if x is None:
        return x

    if dtype is None:
        dtype = infer_type(x)

    if is_tensor_or_variable(x):
        if x.dtype != dtype:
            x = tf.cast(x, dtype=dtype)
        return x
    elif sp.isspmatrix(x):
        return sparse_adj_to_sparse_tensor(x.astype(dtype, copy=False))
    elif isinstance(x, (np.ndarray, np.matrix)) or is_list_like(x) or is_scalar_like(x):
        return tf.convert_to_tensor(x, dtype=dtype)
    else:
        raise TypeError(
            f'Invalid type of inputs data. Allowed data type `(Tensor, SparseTensor, Numpy array, Scipy sparse tensor, None)`, but got {type(x)}.')


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
    if len(xs) > 1:
        return tuple(astensor(x) for x in xs)
    else:
        xs, = xs
        if is_list_like(xs) and not is_scalar_like(xs[0]):
            # Check `not is_scalar_like(xs[0])` to avoid the situation like
            # the list `[1,2,3]` convert to tensor(1), tensor(2), tensor(3)
            return astensors(*xs)
        else:
            return astensor(xs)


def normalize_adj_tensor(adj, rate=-0.5, self_loop=1.0):
    adj = adj + self_loop * tf.eye(tf.shape(adj)[0], dtype=adj.dtype)
    d = tf.reduce_sum(adj, axis=1)
    d_power = tf.pow(d, rate)
    d_power_mat = tf.linalg.diag(d_power)
    return d_power_mat @ adj @ d_power_mat


def add_self_loop_edge(edge_index, edge_weight, n_nodes=None, fill_weight=1.0):

    if n_nodes is None:
        n_nodes = tf.reduce_max(edge_index) + 1

    if edge_weight is None:
        edge_weight = tf.ones([edge_index.shape[0]], dtype=floatx())

    range_arr = tf.range(n_nodes, dtype=intx())
    diagnal_edge_index = tf.stack([range_arr, range_arr], axis=1)
    updated_edge_index = tf.concat([edge_index, diagnal_edge_index], axis=0)

    diagnal_edge_weight = tf.zeros([n_nodes], dtype=floatx()) + fill_weight
    updated_edge_weight = tf.concat([edge_weight, diagnal_edge_weight], axis=0)

    return updated_edge_index, updated_edge_weight


def normalize_edge_tensor(edge_index, edge_weight=None, n_nodes=None, fill_weight=1.0, rate=-0.5):

    if edge_weight is None:
        edge_weight = tf.ones([edge_index.shape[0]], dtype=floatx())

    if n_nodes is None:
        n_nodes = tf.reduce_max(edge_index) + 1

    edge_index, edge_weight = add_self_loop_edge(
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


def sparse_tensor_to_sparse_adj(x):
    """Converts a SparseTensor to a Scipy sparse matrix (CSR matrix)."""
    data = x.values.astype(floatx())
    indices = x.indices.numpy().T
    shape = x.shape
    return sp.csr_matrix((data, indices), shape=shape)
