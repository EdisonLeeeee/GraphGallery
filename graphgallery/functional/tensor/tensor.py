import torch
import warnings
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


def data_type_dict() -> dict:
    return _TYPE


def is_tensor(x: Any) -> bool:
    return torch.is_tensor(x)


def is_sparse(x: Any) -> bool:
    return is_tensor(x) and x.is_sparse


def is_dense(x: Any) -> bool:
    return is_tensor(x) and not x.is_sparse


def astensor(x, *, dtype=None, device=None, escape=None) -> torch.Tensor:

    try:
        if x is None or (escape is not None and isinstance(x, escape)):
            return x
    except TypeError:
        raise TypeError(f"argument 'escape' must be a type or tuple of types.")
    device = torch.device(device) if device is not None else torch.device("cpu")
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

    if isinstance(dtype, (np.dtype, str)):
        dtype = data_type_dict().get(str(dtype), dtype)
    elif not isinstance(dtype, torch.dtype):
        raise TypeError(
            f"argument 'dtype' must be torch.dtype, np.dtype or str, but got {type(dtype)}."
        )

    if is_tensor(x):
        tensor = x.to(dtype)

    elif sp.isspmatrix(x):
        if gg.backend() == "dgl":
            import dgl

            if x.sum() != x.nnz:
                warnings.warn("Got a weighted sparse matrix with elements not equal to 1. "
                              "The element weights can be accessed by `g.edata['_edge_weight'].`")
                tensor = dgl.from_scipy(x, idtype=torch.int64, eweight_name="_edge_weight")
            else:
                tensor = dgl.from_scipy(x, idtype=torch.int64)

        elif gg.backend() == "pyg":
            edge_index, edge_weight = gf.sparse_adj_to_edge(x)
            return (astensor(edge_index,
                             dtype=torch.int64,
                             device=device,
                             escape=escape),
                    astensor(edge_weight,
                             dtype=torch.float32,
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


def astensors(*xs, dtype=None, device=None, escape=None):
    """Convert input matrices to Tensor(s) or SparseTensor(s).

    Parameters:
    ----------
    xs: one or a list of python object(s)
    dtype: The type of Tensor 'x', if not specified,
        it will automatically use appropriate data type.
        See 'graphgallery.infer_type'.
    device: torch.device, optional. the desired device of returned tensor.
        Default: if 'None', uses the CPU device for the default tensor type.     
    escape: a Class or a tuple of Classes, `astensor` will disabled if
        `isinstance(x, escape)`.

    Returns:
    -------     
    Tensor(s) or SparseTensor(s) with dtype. 
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    return _astensors_fn(*xs,
                         dtype=dtype,
                         device=device,
                         escape=escape)


_astensors_fn = gf.multiple(type_check=False)(astensor)


@gf.multiple()
def tensoras(tensor):
    """Convert a TensorFLow tensor or PyTorch Tensor
        to Numpy array or Scipy sparse matrix.
    """

    if is_dense(tensor):
        m = tensor.detach().cpu().numpy()
        if m.ndim == 0:
            m = m.item()
    elif is_sparse(tensor):
        m = gf.sparse_tensor_to_sparse_adj(tensor)
    elif isinstance(tensor, np.ndarray) or sp.isspmatrix(tensor):
        m = tensor.copy()
    else:
        m = np.asarray(tensor)
    return m
