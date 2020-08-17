import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from numbers import Number
from tensorflow.keras import backend as K

from graphgallery import config
from graphgallery.utils.type_check import is_list_like, is_interger_scalar, is_tensor_or_variable, is_scalar_like


__all__ = ['check_and_convert', 'sparse_adj_to_sparse_tensor', 'sparse_tensor_to_sparse_adj',
           'sparse_adj_to_edges', 'edges_to_sparse_adj', 'asintarr', 'astensor', 'astensors',
           ]


def check_and_convert(matrix, is_sparse):
    """Check the input matrix and convert it into a proper data type.

    Arguments:
    ----------
        matrix: Scipy sparse matrix or Numpy array-like or Numpy matrix.
        is_sparse: Indicating whether the input matrix is sparse matrix or not.

    Returns:
    ----------
        A converted matrix with appropriate floating type.

    """
    if is_list_like(matrix):
        return [check_and_convert(m, is_sparse) for m in matrix]

    if not is_sparse:
        if not isinstance(matrix, (np.ndarray, np.matrix)):
            raise TypeError("The input matrix must be Numpy array-like or Numpy matrix"
                            f" when `is_sparse=False`, but got {type(matrix)}")
        return np.asarray(matrix, dtype=config.floatx())
    else:
        if not sp.issparse(matrix):
            raise TypeError(f"The input matrix must be Scipy sparse matrix when `is_sparse=True`, but got {type(matrix)}")

        return matrix.astype(dtype=config.floatx(), copy=False)


def sparse_adj_to_sparse_tensor(x):
    """Converts a Scipy sparse matrix to a SparseTensor."""
    sparse_coo = x.tocoo()
    row, col = sparse_coo.row, sparse_coo.col
    data, shape = sparse_coo.data, sparse_coo.shape
    indices = np.concatenate(
        (np.expand_dims(row, axis=1), np.expand_dims(col, axis=1)), axis=1)
    return tf.sparse.SparseTensor(indices, data, shape)


def sparse_tensor_to_sparse_adj(x):
    """Converts a SparseTensor to a Scipy sparse matrix (CSR matrix)."""
    data = x.values.astype(config.floatx())
    indices = x.indices.numpy().T
    shape = x.shape
    return sp.csr_matrix((data, indices), shape=shape)


def sparse_adj_to_edges(adj):
    """Convert a Scipy sparse matrix to (edge_index, edge_weight) representation

    edge_index: shape [M, 2]
    edge_weight: shape [M,]

    """
    adj = adj.tocoo()
    edge_index = np.stack([adj.row, adj.col], axis=1)
    edge_weight = adj.data

    return edge_index, edge_weight


def edges_to_sparse_adj(edge_index, edge_weight):
    """Convert (edge_index, edge_weight) representation to a Scipy sparse matrix

    edge_index: shape [M, 2]
    edge_weight: shape [M,]

    """
    n = np.max(edge_index) + 1
    edge_index = edge_index.astype('int64', copy=False)
    adj = sp.csr_matrix((edge_weight, (edge_index[:, 0], edge_index[:, 1])), shape=(n, n))
    return adj


def infer_type(x):
    """Infer type of the input `x`.

     Arguments:
    ----------
    x: tf.Tensor, tf.Variable, Scipy sparse matrix, 
        Numpy array-like, etc.

    Returns:
    ----------      
        The converted type of `x`:
        1. `graphgallery.config.floatx()` if `x` is floating
        2. `graphgallery.config.intx() ` if `x` is integer
        3. `Bool` if `x` is bool.

    """

    # For tensor or variable
    if is_tensor_or_variable(x):
        if x.dtype.is_floating:
            return config.floatx()
        elif x.dtype.is_integer or x.dtype.is_unsigned:
            return config.intx()
        elif x.dtype.is_bool:
            return 'bool'
        else:
            raise RuntimeError(f'Invalid input of `{type(x)}`')

    if not hasattr(x, 'dtype'):
        x = np.asarray(x)

    if x.dtype.kind == 'f':
        return config.floatx()
    elif x.dtype.kind == 'i' or x.dtype.kind == 'u':
        return config.intx()
    elif x.dtype.kind == 'b':
        return 'bool'
    else:
        raise RuntimeError(f'Invalid input of `{type(x)}`')


def asintarr(x, dtype=config.intx()):
    """Convert `x` to interger Numpy array.

    Arguments:
    ----------
    x: tf.Tensor, tf.Variable, Scipy sparse matrix, 
        Numpy array-like, etc.

    Returns:
    ----------      
        Integer Numpy array with dtype `graphgallery.config.intx()`

    """
    if is_tensor_or_variable(x):
        if x.dtype != dtype:
            x = tf.cast(x, dtype=dtype)
        return x

    if is_interger_scalar(x):
        x = np.asarray([x], dtype=dtype)
    elif is_list_like(x) or isinstance(x, (np.ndarray, np.matrix)):
        x = np.asarray(x, dtype=dtype)
    else:
        raise TypeError(f'Invalid input which should be either array-like or integer scalar, but got {type(x)}.')
    return x


def astensor(x, dtype=None):
    """Convert input matrices to Tensor or SparseTensor.

    Arguments:
    ----------
    x: tf.Tensor, tf.Variable, Scipy sparse matrix, 
        Numpy array-like, etc.

    dtype: The type of Tensor `x`, if not specified,
        it will automatically using appropriate data type.
        See `graphgallery.infer_type`.

    Returns:
    ----------      
        Tensor or SparseTensor with dtype:       
        1. `graphgallery.config.floatx()` if `x` is floating
        2. `graphgallery.config.intx() ` if `x` is integer
        3. `Bool` if `x` is bool.
    """

    if x is None:
        return x

    if dtype is None:
        dtype = infer_type(x)

    if is_tensor_or_variable(x) or K.is_sparse(x):
        if x.dtype != dtype:
            x = tf.cast(x, dtype=dtype)
        return x
    elif sp.isspmatrix(x):
        return sparse_adj_to_sparse_tensor(x.astype(dtype, copy=False))
    elif isinstance(x, (np.ndarray, np.matrix)) or is_list_like(x) or is_scalar_like(x):
        return tf.convert_to_tensor(x, dtype=dtype)
    else:
        raise TypeError(f'Invalid type of inputs data. Allowed data type `(Tensor, SparseTensor, Numpy array, Scipy sparse tensor, None)`, but got {type(x)}.')


def astensors(xs):
    """Convert input matrices to Tensor(s) or SparseTensor(s).

    Arguments:
    ----------
    xs: tf.Tensor, tf.Variable, Scipy sparse matrix, 
        Numpy array-like, or a list of them, etc.

    Returns:
    ----------      
        Tensor(s) or SparseTensor(s) with dtype:       
        1. `graphgallery.config.floatx()` if `x` in `xs` is floating
        2. `graphgallery.config.intx() ` if `x` in `xs` is integer
        3. `Bool` if `x` in `xs` is bool.
    """
    # Check `not isinstance(xs[0], Number)` to avoid the situation like [1,2,3],
    # where [1,2,3] will be converted to three tensors seperately.
    if is_list_like(xs) and not isinstance(xs[0], Number):
        return [astensors(x) for x in xs]
    else:
        return astensor(xs)
