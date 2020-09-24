import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from graphgallery import floatx, intx
from graphgallery.utils.type_check import (is_list_like,
                                           is_interger_scalar,
                                           is_tensor_or_variable,
                                           is_scalar_like)

from graphgallery.utils.decorators import MultiInputs
from graphgallery import transformers as T



def sparse_edges_to_sparse_tensor(edge_index: np.ndarray, edge_weight: np.ndarray = None, shape: tuple = None):
    """
    edge_index: shape [2, M]
    edge_weight: shape [M,]
    """
    if edge_weight is None:
        edge_weight = tf.ones(edge_index.shape[1], dtype=floatx())

    if shape is None:
        N = np.max(edge_index) + 1
        shape = (N, N)

    return tf.SparseTensor(edge_index.T, edge_weight, shape)


def sparse_adj_to_sparse_tensor(x: sp.csr_matrix, dtype=None):
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
    S: tf.sparse.SparseTensor
        Matrix as a sparse tensor.
    """

    if isinstance(dtype, tf.dtypes.DType):
        dtype = dtype.name
    elif dtype is None:
        dtype = infer_type(x)

    edge_index, edge_weight = T.sparse_adj_to_sparse_edges(x)
    return sparse_edges_to_sparse_tensor(edge_index, edge_weight.astype(dtype, copy=False), x.shape)


def sparse_tensor_to_sparse_adj(x):
    """Converts a SparseTensor to a Scipy sparse matrix (CSR matrix)."""
    data = x.values.numpy()
    indices = x.indices.numpy().T
    shape = x.shape
    return sp.csr_matrix((data, indices), shape=shape)


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
        3. `'bool'` if `x` is bool.
    """

    if x is None:
        return x

    if dtype is None:
        dtype = infer_type(x)
    elif isinstance(dtype, str):
        ...
        # TODO
    elif isinstance(dtype, tf.dtypes.DType):
        dtype = dtype.name
    else:
        raise TypeError(
            f"argument 'dtype' must be tensorflow.dtypes.DType or str, not {type(dtype).__name__}.")

    if is_tensor_or_variable(x):
        if x.dtype != dtype:
            x = tf.cast(x, dtype=dtype)
        return x
    elif sp.isspmatrix(x):
        return sparse_adj_to_sparse_tensor(x, dtype=dtype)
    elif isinstance(x, (np.ndarray, np.matrix)) or is_list_like(x) or is_scalar_like(x):
        return tf.convert_to_tensor(x, dtype=dtype)
    else:
        raise TypeError(
            f'Invalid type of inputs data. Allowed data type `(Tensor, SparseTensor, Numpy array, Scipy sparse tensor, None)`, but got {type(x)}.')


astensors = MultiInputs(type_check=False)(astensor)
astensors.__doc__ = """Convert input matrices to Tensor(s) or SparseTensor(s).
    Parameters:
    ----------
    xs: tf.Tensor, tf.Variable, Scipy sparse matrix, 
        Numpy array-like, or a list of them, etc.
        
    dtype: The type of Tensor for all tensors in `xs`, if not specified,
        it will automatically using appropriate data type.
        See `graphgallery.infer_type`.
        
    Returns:
    ----------      
    Tensor(s) or SparseTensor(s) with dtype:       
        1. `graphgallery.floatx()` if `x` in `xs` is floating
        2. `graphgallery.intx() ` if `x` in `xs` is integer
        3. `'bool'` if `x` in 'xs' is bool.
    """


def normalize_adj_tensor(adj, rate=-0.5, fill_weight=1.0):
    if fill_weight:
        adj = adj + fill_weight * tf.eye(tf.shape(adj)[0], dtype=adj.dtype)
    d = tf.reduce_sum(adj, axis=1)
    d_power = tf.pow(d, rate)
    d_power_mat = tf.linalg.diag(d_power)
    return d_power_mat @ adj @ d_power_mat


def add_selfloops_edge(edge_index, edge_weight, n_nodes=None, fill_weight=1.0):

    if n_nodes is None:
        n_nodes = tf.reduce_max(edge_index) + 1

    if edge_weight is None:
        edge_weight = tf.ones([edge_index.shape[0]], dtype=floatx())

    range_arr = tf.range(n_nodes, dtype=edge_index.dtype)
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
