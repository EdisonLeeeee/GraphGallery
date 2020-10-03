import torch
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from collections import Iterable

from graphgallery import backend

__all__ = ['is_list_like', 'is_tf_sparse_tensor',
           'is_th_sparse_tensor', 'is_sparse_tensor',
           'is_scalar_like',
           'is_tf_tensor', 'is_th_tensor',
           'is_tensor',
           'is_interger_scalar']

def is_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def is_list_like(x):
    """Check whether `x` is list like, e.g., Tuple or List.

    Parameters:
        x: A python object to check.

    Returns:
        `True` iff `x` is a list like sequence.
    """
    return isinstance(x, (list, tuple))


def is_tf_sparse_tensor(x):
    """Check whether `x` is a sparse Tensor.

    Check whether an object is a `tf.sparse.SparseTensor`.

    NOTE: This method is different with `scipy.sparse.is_sparse`
    which checks whether `x` is Scipy sparse matrix.

    Parameters:
        x: A python object to check.

    Returns:
        `True` iff `x` is a `tf.sparse.SparseTensor`.
    """
    return K.is_sparse(x)

def is_th_sparse_tensor(x):
    """Check whether `x` is a sparse Tensor.

    Check whether an object is a `torch.sparse.Tensor`.

    NOTE: This method is different with `scipy.sparse.is_sparse`
    which checks whether `x` is Scipy sparse matrix.

    Parameters:
        x: A python object to check.

    Returns:
        `True` iff `x` is a `torch.sparse.Tensor (COO Tensor)`.
    """
    # TODO: is it right?
    return isinstance(x, torch.Tensor) and x.layout != torch.strided

def is_sparse_tensor(x):
    """Check whether `x` is a sparse Tensor."""

    if backend().kind == "T":
        return is_tf_sparse_tensor(x)
    else:
        return is_th_sparse_tensor(x)

def is_tf_tensor(x):
    return any((tf.is_tensor(x),
                    isinstance(x, tf.Variable),
                    isinstance(x, tf.RaggedTensor),
                    is_tf_sparse_tensor(x)))

def is_th_tensor(x):
    # TODO: is it really work for all torch tensors?? maybe work for variable, parameters?
    return torch.is_tensor(x)

def is_tensor(x):
    """Check whether `x` is 
        tf.Tensor,
        tf.Variable,
        tf.RaggedTensor,
        tf.sparse.SparseTensor,
        torch.Tensor, 
        torch.sparse.Tensor.

    Parameters:
        x: A python object to check.

    Returns:
        `True` iff `x` is a (tf or torch) (sparse-)tensor.
    """
    if backend().kind == "T":
        return is_tf_tensor(x)
    else:
        return is_th_tensor(x)

def is_interger_scalar(x):
    """Check whether `x` is an Integer scalar.

    Parameters:
        x: A python object to check.

    Returns:
        `True` iff `x` is a Integer scalar (built-in or Numpy integer).
    """
    return isinstance(x, (int, np.int8,
                          np.int16,
                          np.int32,
                          np.int64,
                          np.uint8,
                          np.uint16,
                          np.uint32,
                          np.uint64,
                          ))


def is_scalar_like(x):
    """Check whether `x` is a scalar, an array scalar, or a 0-dim array.

    Parameters:
        x: A python object to check.

    Returns:
        `True` iff `x` is a scalar, an array scalar, or a 0-dim array.
    """
    return np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
