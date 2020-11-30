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

    try:
        if x is None or (escape is not None and isinstance(x, escape)):
            return x
    except TypeError:
        raise TypeError(f"argument 'escape' must be a type or tuple of types.")

    if dtype is None:
        dtype = gf.infer_type(x)
    elif isinstance(dtype, (np.dtype, str)):
        dtype = data_type_dict().get(str(dtype), dtype)
    elif not isinstance(dtype, torch.dtype):
        raise TypeError(
            f"argument 'dtype' must be torch.dtype, np.dtype or str, but got {type(dtype)}."
        )

    if is_tensor(x):
        tensor = x.to(dtype)
    elif gf.is_tensor(x, backend='tensorflow'):
        return astensor(gf.tensoras(x),
                        dtype=dtype,
                        device=device,
                        escape=escape)
    elif sp.isspmatrix(x):
        if gg.backend() == "dgl_torch":
            import dgl
            tensor = dgl.from_scipy(x, idtype=getattr(torch, gg.intx()))
        elif gg.backend() == "pyg":
            edge_index, edge_weight = gf.sparse_adj_to_edge(x)
            return (astensor(edge_index,
                             dtype=gg.intx(),
                             device=device,
                             escape=escape),
                    astensor(edge_weight,
                             dtype=gg.floatx(),
                             device=device,
                             escape=escape))
        else:
            tensor = sparse_adj_to_sparse_tensor(x, dtype=dtype)
    elif any((isinstance(x, (np.ndarray, np.matrix)), gg.is_listlike(x),
              gg.is_scalar(x))):
        tensor = torch.tensor(x, dtype=dtype, device=device)
    else:
        raise TypeError(
            f"Invalid type of inputs. Allowed data type (Tensor, SparseTensor, Numpy array, Scipy sparse tensor, None), but got {type(x)}."
        )
    return tensor.to(device)
