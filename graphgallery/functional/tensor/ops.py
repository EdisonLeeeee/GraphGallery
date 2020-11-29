import torch
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from typing import Any

import graphgallery as gg
from graphgallery import functional as F

from . import tensorflow
from . import pytorch


def infer_type(x: Any) -> str:
    """Infer type of the input 'x'.

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
            raise RuntimeError(f"Invalid input of '{type(x)}'")

    elif tensorflow.is_tensor(x):
        if x.dtype.is_floating:
            return gg.floatx()
        elif x.dtype.is_integer or x.dtype.is_unsigned:
            return gg.intx()
        elif x.dtype.is_bool:
            return gg.boolx()
        else:
            raise RuntimeError(f"Invalid input of '{type(x)}'")

    if not hasattr(x, 'dtype'):
        x = np.asarray(x)

    if x.dtype.kind in {'f', 'c'}:
        return gg.floatx()
    elif x.dtype.kind in {'i', 'u'}:
        return gg.intx()
    elif x.dtype.kind == 'b':
        return gg.boolx()
    elif x.dtype.kind == 'O':
        raise RuntimeError(f"Invalid inputs of '{x}'.")
    else:
        raise RuntimeError(f"Invalid input of '{type(x)}'.")


def sparse_adj_to_sparse_tensor(x, backend=None):
    """Converts a Scipy sparse matrix to a TensorFlow/PyTorch SparseTensor.

    Parameters
    ----------
    x: Scipy sparse matrix
        Matrix in Scipy sparse format.

    backend: String or 'BackendModule', optional.
     `'tensorflow'`, `'torch'`, TensorFlowBackend, PyTorchBackend, etc.
     if not specified, return the current default backend module. 

    Returns
    -------
    S: SparseTensor
        Matrix as a sparse tensor.
    """
    backend = gg.backend(backend)

    if backend == "tensorflow":
        return tensorflow.sparse_adj_to_sparse_tensor(x)
    else:
        return pytorch.sparse_adj_to_sparse_tensor(x)


def sparse_tensor_to_sparse_adj(x, *, backend=None):
    """Converts a SparseTensor to a Scipy sparse matrix (CSR matrix)."""
    backend = gg.backend(backend)

    if backend == "tensorflow":
        return tensorflow.sparse_tensor_to_sparse_adj(x)
    else:
        return pytorch.sparse_tensor_to_sparse_adj(x)


def sparse_edge_to_sparse_tensor(edge_index: np.ndarray,
                                 edge_weight: np.ndarray = None,
                                 shape: tuple = None,
                                 backend=None):

    backend = gg.backend(backend)
    if backend == "tensorflow":
        return tensorflow.sparse_edge_to_sparse_tensor(edge_index, edge_weight,
                                                       shape)
    else:
        return pytorch.sparse_edge_to_sparse_tensor(edge_index, edge_weight,
                                                    shape)


#### only works for tensorflow backend now #####################################


def normalize_adj_tensor(adj, rate=-0.5, fill_weight=1.0, backend=None):
    backend = gg.backend(backend)
    if backend == "tensorflow":
        return tensorflow.normalize_adj_tensor(adj,
                                               rate=rate,
                                               fill_weight=fill_weight)
    else:
        # TODO
        return pytorch.normalize_adj_tensor(adj,
                                            rate=rate,
                                            fill_weight=fill_weight)


def add_selfloops_edge(edge_index,
                       edge_weight,
                       num_nodes=None,
                       fill_weight=1.0,
                       backend=None):
    backend = gg.backend(backend)
    if backend == "tensorflow":
        return tensorflow.normalize_adj_tensor(edge_index,
                                               edge_weight,
                                               num_nodes=num_nodes,
                                               fill_weight=fill_weight)
    else:
        # TODO
        return pytorch.normalize_adj_tensor(edge_index,
                                            edge_weight,
                                            num_nodes=num_nodes,
                                            fill_weight=fill_weight)


def normalize_edge_tensor(edge_index,
                          edge_weight=None,
                          num_nodes=None,
                          fill_weight=1.0,
                          rate=-0.5,
                          backend=None):
    backend = gg.backend(backend)
    if backend == "tensorflow":
        return tensorflow.normalize_adj_tensor(edge_index,
                                               edge_weight=edge_weight,
                                               num_nodes=num_nodes,
                                               fill_weight=fill_weight,
                                               rate=rate)
    else:
        # TODO
        return pytorch.normalize_adj_tensor(edge_index,
                                            edge_weight=edge_weight,
                                            num_nodes=num_nodes,
                                            fill_weight=fill_weight,
                                            rate=rate)
