import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import graphgallery as gg
import tensorflow.keras.backend as K

from typing import Any
from graphgallery import functional as F
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


def is_sparse_(x: Any) -> bool:
    return K.is_sparse(x)


def is_dense(x: Any) -> bool:
    return any((isinstance(x, tf.Tensor), isinstance(x, tf.Variable),
                isinstance(x, tf.RaggedTensor)))


def is_tensor(x: Any) -> bool:
    return is_dense(x) or is_sparse_(x)


def astensor(x, *, dtype=None, device=None, escape=None):

    if x is None:
        return x

    if escape is not None and isinstance(x, escape):
        return x

    if dtype is None:
        from ..ops import infer_type
        dtype = infer_type(x)
    elif isinstance(dtype, str):
        ...
        # TODO
    elif isinstance(dtype, tf.dtypes.DType):
        dtype = dtype.name
    else:
        raise TypeError(
            f"argument 'dtype' must be tensorflow.dtypes.DType or str, not {type(dtype).__name__}."
        )

    with tf.device(device):
        if is_tensor(x):
            if x.dtype != dtype:
                x = tf.cast(x, dtype=dtype)
            return x
        # TODO
        # elif gg.is_tensor(x, backend='torch'):
        #     from ..tensor import tensoras
        #     return astensor(tensoras(x),
        #                     dtype=dtype,
        #                     device=device,
        #                     escape=escape)
        elif sp.isspmatrix(x):
            if gg.backend() == "dgl_tf":
                try:
                    import dgl
                    return dgl.from_scipy(x,
                                          idtype=getattr(tf,
                                                         gg.intx())).to(device)
                except ImportError:
                    return sparse_adj_to_sparse_tensor(x, dtype=dtype)
            else:
                return sparse_adj_to_sparse_tensor(x, dtype=dtype)
        elif isinstance(
                x,
            (np.ndarray, np.matrix)) or gg.is_listlike(x) or gg.is_scalar(x):
            return tf.convert_to_tensor(x, dtype=dtype)
        else:
            raise TypeError(
                f'Invalid type of inputs data. Allowed data type (Tensor, SparseTensor, Numpy array, Scipy sparse matrix, None), but got {type(x)}.'
            )
