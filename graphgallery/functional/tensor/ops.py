import torch
import numpy as np
import scipy.sparse as sp
from typing import Any, Union, Optional

import graphgallery as gg
from graphgallery import functional as gf

from . import pytorch


def get_module(backend=None):
    """get the module of eigher
    'graphgallery.functional.tensor.tensorflow'
    or 'graphgallery.functional.tensor.pytorch'
    by 'backend'.

    Parameters
    ----------
    backend: String or BackendModule, optional.
        'tensorflow', 'torch', TensorFlowBackend,
        PyTorchBackend, etc. if not specified,
        return the current backend module.

    Returns
    -------
    module:
    - 'graphgallery.functional.tensor.tensorflow'
        for tensorflow backend,
    - 'graphgallery.functional.tensor.pytorch'
        for pytorch backend
    """
    backend = gg.backend(backend)

    if backend == "tensorflow":
        from . import tensorflow
        return tensorflow
    else:
        return pytorch


def infer_type(x: Any) -> str:
    """Infer the type of the input 'x'.

    Parameters:
    ----------
    x: Any python object

    Returns:
    ----------
    dtype: string, the proper data type of 'x':
        1. 'graphgallery.floatx()' if 'x' is floating,
        2. 'graphgallery.intx()' if 'x' is integer,
        3. 'graphgallery.boolx()' if 'x' is boolean.
    """
    # For tensor or variable
    if pytorch.is_tensor(x):
        if x.dtype.is_floating_point:
            return gg.floatx()
        elif x.dtype == torch.bool:
            return gg.boolx()
        elif 'int' in str(x.dtype):
            return gg.intx()
        else:
            raise TypeError(f"Invalid type of pytorch Tensor: '{type(x)}'")

    # else:
    #     try:
    #         from . import tensorflow
    #         if tensorflow.is_tensor(x):
    #             if x.dtype.is_floating:
    #                 return gg.floatx()
    #             elif x.dtype.is_integer or x.dtype.is_unsigned:
    #                 return gg.intx()
    #             elif x.dtype.is_bool:
    #                 return gg.boolx()
    #             else:
    #                 raise TypeError(f"Invalid type of tensorflow Tensor: '{type(x)}'")
    #     except (ImportError, ModuleNotFoundError):
    #         pass

    _x = x
    if not hasattr(_x, 'dtype'):
        _x = np.asarray(_x)

    if _x.dtype.kind in {'f', 'c'}:
        return gg.floatx()
    elif _x.dtype.kind in {'i', 'u'}:
        return gg.intx()
    elif _x.dtype.kind == 'b':
        return gg.boolx()
    elif _x.dtype.kind == 'O':
        raise TypeError(f"Invalid inputs of '{x}'.")
    else:
        raise TypeError(f"Invalid input of '{type(x).__name__}'.")


def gather(x, index_or_mask=None, *, backend=None):
    # TODO axis?
    module = get_module(backend)
    return module.gather(x, index_or_mask)


def to(tensor, device_or_dtype):
    # TODO
    raise NotImplemented


def sparse_adj_to_sparse_tensor(x, *, backend=None):
    """Converts a Scipy sparse matrix to a TensorFlow/PyTorch SparseTensor.

    Parameters
    ----------
    x: Scipy sparse matrix
        Matrix in Scipy sparse format.
    backend: String or BackendModule, optional.
        'tensorflow', 'torch', TensorFlowBackend, 
        PyTorchBackend, etc. if not specified, 
        return the current backend module. 

    Returns
    -------
    S: SparseTensor
        Matrix as a tensorflow/pytorch sparse tensor.
    """
    module = get_module(backend)
    return module.sparse_adj_to_sparse_tensor(x)


def sparse_tensor_to_sparse_adj(x, *, backend=None):
    """Converts a SparseTensor to a Scipy sparse matrix (CSR matrix)."""
    module = get_module(backend)
    return module.sparse_tensor_to_sparse_adj(x)


def sparse_edge_to_sparse_tensor(edge_index: np.ndarray,
                                 edge_weight: np.ndarray = None,
                                 shape: tuple = None,
                                 backend=None):
    module = get_module(backend)
    return module.sparse_edge_to_sparse_tensor(edge_index, edge_weight, shape)


#### only works for tensorflow backend now #####################################
def normalize_adj_tensor(adj,
                         rate=-0.5,
                         fill_weight=1.0,
                         backend=None):
    module = get_module(backend)
    return module.normalize_adj_tensor(adj, rate=rate, fill_weight=fill_weight)


def add_selfloops_edge_tensor(edge_index,
                              edge_weight,
                              num_nodes=None,
                              fill_weight=1.0,
                              backend=None):
    module = get_module(backend)
    return module.add_selfloops_edge_tensor(edge_index,
                                            edge_weight,
                                            num_nodes=num_nodes,
                                            fill_weight=fill_weight)


def normalize_edge_tensor(edge_index,
                          edge_weight=None,
                          num_nodes=None,
                          fill_weight=1.0,
                          rate=-0.5,
                          backend=None):
    module = get_module(backend)
    return module.normalize_edge_tensor(edge_index,
                                        edge_weight,
                                        num_nodes=num_nodes,
                                        fill_weight=fill_weight,
                                        rate=rate)
