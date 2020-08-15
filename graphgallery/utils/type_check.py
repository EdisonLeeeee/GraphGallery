
import tensorflow as tf
import numpy as np
import collections.abc as collections_abc


def is_list_like(x):
    """Check whether `x` is list like, e.g., Tuple or List.

    Arguments:
        x: A python object to check.

    Returns:
        `True` iff `x` is a list like sequence.
    """
    return isinstance(x, collections_abc.Sequence)


def is_sparse(x):
    """Check whether `x` is sparse Tensor.

    Check whether an object is a `tf.sparse.SparseTensor` or
    `tf.compat.v1.SparseTensorValue`.

    NOTE: This method is different with `scipy.sparse.is_sparse`
    which is checking  whether `x` is Scipy sparse matrix.

    Arguments:
        x: A python object to check.

    Returns:
        `True` iff `x` is a `tf.sparse.SparseTensor` or
        `tf.compat.v1.SparseTensorValue`.
    """
    return isinstance(x, (tf.sparse.SparseTensor, tf.sparse.SparseTensorValue))


def is_tensor_or_variable(x):
    """Check whether `x` is tf.Tensor or tf.Variable or tf.RaggedTensor.

    Arguments:
        x: A python object to check.

    Returns:
        `True` iff `x` is a `tf.Tensor` or `tf.Variable` or `tf.RaggedTensor`.
    """
    return tf.is_tensor(x) or isinstance(x, tf.Variable) or isinstance(x, tf.RaggedTensor)


def is_interger_scalar(x):
    """Check whether `x` is an Integer scalar.

    Arguments:
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

    Arguments:
        x: A python object to check.

    Returns:
        `True` iff `x` is a scalar, an array scalar, or a 0-dim array.
    """
    return np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
