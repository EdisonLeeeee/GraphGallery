import torch
import numpy as np
import tensorflow as tf
import graphgallery as gg
import tensorflow.keras.backend as K

from collections import Iterable
from typing import Any, Optional


__all__ = ['is_iterable',
           'is_listlike',
           'is_scalar',
           'is_intscalar',
           'infer_type',
           'is_tensor',
           'is_strided_tensor',
           'is_sparse',
           ]


def is_iterable(obj: Any) -> bool:
    """Check whether `x` is an iterable object except for string."""
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def is_listlike(x: Any) -> bool:
    """Check whether `x` is list like, e.g., Tuple or List.
    Parameters:
    ----------
    x: A python object to check.
    Returns:
    ----------
    `True` iff `x` is a list like sequence.
    """
    return isinstance(obj, (list, tuple)) or (isinstance(obj, np.ndarray) and obj.dtype == "O")


def is_scalar(x: Any) -> bool:
    """Check whether `x` is a scalar, an array scalar, or a 0-dim array.
    Parameters:
    ----------
    x: A python object to check.
    Returns:
    ----------
    `True` iff `x` is a scalar, an array scalar, or a 0-dim array.
    """
    return np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)


def is_intscalar(x: Any) -> bool:
    """Check whether `x` is an integer scalar.
    Parameters:
    ----------
    x: A python object to check.
    Returns:
    ----------
    `True` iff `x` is an integer scalar (built-in or Numpy integer).
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


def is_floatscalar(x: Any) -> bool:
    """Check whether `x` is a float scalar.
    Parameters:
    ----------
    x: A python object to check.
    Returns:
    ----------
    `True` iff `x` is a float scalar (built-in or Numpy float).
    """
    return isinstance(x, (float,
                          np.float16,
                          np.float32,
                          np.float64,
                          ))


def infer_type(x: Any) -> str:
    """Infer type of the input `x`.
    Parameters:
    ----------
    x: Any python object
    Returns:
    ----------
    dtype: string, the proper data type of `x`:
        1. `graphgallery.floatx()` if `x` is floating,
        2. `graphgallery.intx()` if `x` is integer,
        3. `graphgallery.boolx()` if `x` is boolean.
    """
    # For tensor or variable
    if is_th_tensor(x):
        if x.dtype.is_floating_point:
            return gg.floatx()
        elif x.dtype == torch.bool:
            return gg.boolx()
        elif 'int' in str(x.dtype):
            return gg.intx()
        else:
            raise RuntimeError(f'Invalid input of `{type(x)}`')

    elif is_tf_tensor(x):
        if x.dtype.is_floating:
            return gg.floatx()
        elif x.dtype.is_integer or x.dtype.is_unsigned:
            return gg.intx()
        elif x.dtype.is_bool:
            return gg.boolx()
        else:
            raise RuntimeError(f'Invalid input of `{type(x)}`')

    if not hasattr(x, 'dtype'):
        x = np.asarray(x)

    if x.dtype.kind in {'f', 'c'}:
        return gg.floatx()
    elif x.dtype.kind in {'i', 'u'}:
        return gg.intx()
    elif x.dtype.kind == 'b':
        return gg.boolx()
    elif x.dtype.kind == 'O':
        raise RuntimeError(f'Invalid inputs of `{x}`.')
    else:
        raise RuntimeError(f'Invalid input of `{type(x)}`')


def is_sparse(x: Any, backend: Optional[str] = None) -> bool:
    """Check whether `x` is a sparse Tensor.

    Parameters:
    ----------
    x: A python object to check.

    backend: String or 'BackendModule', optional.    
     `'tensorflow'`, `'torch'`, TensorFlowBackend, PyTorchBackend, etc.    
     if not specified, return the current default backend module. 
    Returns:
    ----------
    `True` iff `x` is a (tf or torch) sparse-tensor.
    """
    backend = gg.backend(backend)

    if backend == 'tensorflow':
        return is_tf_sparse_tensor(x)
    else:
        return is_th_sparse_tensor(x)


def is_strided_tensor(x: Any, backend: Optional[str] = None) -> bool:
    """Check whether `x` is a strided (dense) Tensor.

    Parameters:
    ----------
    x: A python object to check.

    backend: String or 'BackendModule', optional.    
     `'tensorflow'`, `'torch'`, TensorFlowBackend, PyTorchBackend, etc.    
     if not specified, return the current default backend module. 

    Returns:
    ----------
    `True` iff `x` is a (tf or torch) strided (dense) Tensor.
    """

    backend = gg.backend(backend)

    if backend == 'tensorflow':
        return is_tf_strided_tensor(x)
    else:
        return is_th_strided_tensor(x)


def is_tensor(x: Any, backend: Optional[str] = None) -> bool:
    """Check whether `x` is 
        tf.Tensor,
        tf.Variable,
        tf.RaggedTensor,
        tf.sparse.SparseTensor,
        torch.Tensor, 
        torch.sparse.Tensor.
    Parameters:
    ----------
    x: A python object to check.

    backend: String or 'BackendModule', optional.    
     `'tensorflow'`, `'torch'`, TensorFlowBackend, PyTorchBackend, etc.    
     if not specified, return the current default backend module.    
    Returns:
    ----------
    `True` iff `x` is a (tf or torch) (sparse-)tensor.
    """
    backend = gg.backend(backend)

    if backend == 'tensorflow':
        return is_tf_tensor(x)
    else:
        return is_th_tensor(x)


def is_tf_sparse_tensor(x: Any) -> bool:
    return K.is_sparse(x)


def is_th_sparse_tensor(x: Any) -> bool:
    return is_th_tensor(x) and not is_th_strided_tensor(x)


def is_tf_strided_tensor(x: Any) -> bool:
    return any((isinstance(x, tf.Tensor),
                isinstance(x, tf.Variable),
                isinstance(x, tf.RaggedTensor)))


def is_th_strided_tensor(x: Any) -> bool:
    return is_th_tensor(x) and x.layout == torch.strided


def is_tf_tensor(x: Any) -> bool:
    return is_tf_strided_tensor(x) or is_tf_sparse_tensor(x)


def is_th_tensor(x: Any) -> bool:
    return torch.is_tensor(x)
