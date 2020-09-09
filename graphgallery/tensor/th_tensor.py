import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from graphgallery import floatx, intx
from graphgallery.utils.type_check import (is_list_like,
                                           is_interger_scalar,
                                           is_tensor_or_variable,
                                           is_scalar_like)


from graphgallery import gallery_export


def sparse_adj_to_sparse_tensor(x):
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


@gallery_export("graphgallery.asthtensor")
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

    ...


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


def normalize_adj_tensor(adj, rate=-0.5, selfloop=1.0):
    ...


def add_selfloop_edge(edge_index, edge_weight, n_nodes=None, fill_weight=1.0):

    ...


def normalize_edge_tensor(edge_index, edge_weight=None, n_nodes=None, fill_weight=1.0, rate=-0.5):

    ...


def sparse_tensor_to_sparse_adj(x):
    """Converts a SparseTensor to a Scipy sparse matrix (CSR matrix)."""
    ...
