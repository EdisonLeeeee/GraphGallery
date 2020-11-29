import torch
import numpy as np
import scipy.sparse as sp
import graphgallery as gg
from typing import Any

from graphgallery import functional as gf
from .ops import sparse_adj_to_sparse_tensor

_TYPE = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
    'uint8': torch.uint8,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
    'bool': torch.bool
}


def data_type_dict():
    return _TYPE


def is_tensor(x: Any) -> bool:
    return torch.is_tensor(x)


def is_sparse(x: Any) -> bool:
    return is_tensor(x) and not is_dense(x)


def is_dense(x: Any) -> bool:
    return is_tensor(x) and x.layout == torch.strided


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
    elif isinstance(dtype, torch.dtype):
        dtype = str(dtype).split('.')[-1]
    else:
        raise TypeError(
            f"argument 'dtype' must be torch.dtype or str, not {type(dtype).__name__}."
        )

    if is_tensor(x):
        tensor = x.to(getattr(torch, dtype))
    # TODO
    # elif gg.is_tensor(x, backend='tensorflow'):
    #     from ..tensor import tensoras
    #     return astensor(tensoras(x), dtype=dtype, device=device, escape=escape)
    elif sp.isspmatrix(x):
        if gg.backend() == "dgl_torch":
            try:
                import dgl
                tensor = dgl.from_scipy(x, idtype=getattr(torch, gg.intx()))
            except ImportError:
                tensor = sparse_adj_to_sparse_tensor(x, dtype=dtype)
        else:
            tensor = sparse_adj_to_sparse_tensor(x, dtype=dtype)

    elif isinstance(
        x,
            (np.ndarray, np.matrix)) or gg.is_listlike(x) or gg.is_scalar(x):
        tensor = torch.tensor(x, dtype=getattr(torch, dtype), device=device)
    else:
        raise TypeError(
            f'Invalid type of inputs data. Allowed data type `(Tensor, SparseTensor, Numpy array, Scipy sparse tensor, None)`, but got {type(x)}.'
        )
    return tensor.to(device)
