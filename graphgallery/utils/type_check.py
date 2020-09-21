import torch
import numpy as np
import tensorflow as tf
import collections.abc as collections_abc
import tensorflow.keras.backend as K

from graphgallery import backend


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


def is_tensor_or_variable(x):
    """Check whether `x` is tf.Tensor or tf.Variable or tf.RaggedTensor.

    Parameters:
        x: A python object to check.

    Returns:
        `True` iff `x` is a `tf.Tensor` or `tf.Variable` or `tf.RaggedTensor`.
    """
    if backend().kind == "T":
        return any((tf.is_tensor(x),
                    isinstance(x, tf.Variable),
                    isinstance(x, tf.RaggedTensor),
                    is_tf_sparse_tensor(x)))
    else:
        # TODO: is it really work for all torch tensors??
        return torch.is_tensor(x)


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
