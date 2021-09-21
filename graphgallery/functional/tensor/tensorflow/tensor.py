import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import graphgallery as gg

from typing import Any
from graphgallery import functional as gf
from .ops import sparse_adj_to_sparse_tensor

_TYPE = {
    'float16': tf.float16,
    'float32': tf.float32,
    'float64': tf.float64,
    'uint8': tf.uint8,
    'int8': tf.int8,
    'int16': tf.int16,
    'int32': tf.int32,
    'int64': tf.int64,
    'bool': tf.bool
}


def data_type_dict():
    return _TYPE


def is_sparse(x: Any) -> bool:
    return tf.keras.backend.is_sparse(x)


def is_dense(x: Any) -> bool:
    # is 'RaggedTensor' a dense tensor?
    return any((isinstance(x, tf.Tensor), isinstance(x, tf.Variable),
                isinstance(x, tf.RaggedTensor)))


def is_tensor(x: Any) -> bool:
    return is_dense(x) or is_sparse(x)


def astensor(x, *, dtype=None, device=None, escape=None):

    try:
        if x is None or (escape is not None and isinstance(x, escape)):
            return x
    except TypeError:
        raise TypeError(f"argument 'escape' must be a type or tuple of types.")

    # update: accept `dict` instance
    if isinstance(x, dict):
        for k, v in x.items():
            try:
                x[k] = astensor(v, dtype=dtype, device=device, escape=escape)
            except TypeError:
                pass
        return x

    if dtype is None:
        dtype = gf.infer_type(x)
    elif isinstance(dtype, tf.dtypes.DType):
        dtype = dtype.name
    elif isinstance(dtype, (np.dtype, str)):
        dtype = str(dtype)
    else:
        raise TypeError(
            f"argument 'dtype' must be tf.dtypes.DType, np.dtype or str, but got {type(dtype)}."
        )

    with tf.device(device):
        if is_tensor(x):
            if x.dtype != dtype:
                return tf.cast(x, dtype=dtype)
            return tf.identity(x)
        elif gf.is_tensor(x, backend='torch'):
            return astensor(gf.tensoras(x),
                            dtype=dtype,
                            device=device,
                            escape=escape)
        elif sp.isspmatrix(x):
            return sparse_adj_to_sparse_tensor(x, dtype=dtype)
        elif any((isinstance(x, (np.ndarray, np.matrix)), gg.is_listlike(x),
                  gg.is_scalar(x))):
            return tf.convert_to_tensor(x, dtype=dtype)
        else:
            raise TypeError(
                f"Invalid type of inputs. Allowed data type(Tensor, SparseTensor, Numpy array, Scipy sparse matrix, None), but got {type(x)}."
            )
