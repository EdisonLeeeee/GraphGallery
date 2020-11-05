import torch
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

import graphgallery as gg

from .device import parse_device
from .tensorflow import tensor as tf_tensor
from .pytorch import tensor as th_tensor
from ..decorators import MultiInputs


__all__ = ["astensor", "astensors", "tensoras", "tensor2tensor",
           "sparse_adj_to_sparse_tensor",
           "sparse_tensor_to_sparse_adj",
           "sparse_edge_to_sparse_tensor",
           "normalize_adj_tensor",
           "add_selfloops_edge",
           "normalize_edge_tensor"]


def astensor(x, *, dtype=None, device=None, backend=None, escape=None):
    """Convert input object to Tensor or SparseTensor.

    Parameters:
    ----------
    x: any python object

    dtype: The type of Tensor `x`, if not specified,
        it will automatically use appropriate data type.
        See `graphgallery.infer_type`.
    device: tf.device, optional. the desired device of returned tensor.
        Default: if `None`, uses the CPU device for the default tensor type.
    backend: String or 'BackendModule', optional.
         `'tensorflow'`, `'torch'`, TensorFlowBackend, PyTorchBackend, etc.
         if not specified, return the current default backend module. 
    escape: a Class or a tuple of Classes,  `astensor` will disabled if
         `isinstance(x, escape)`.

    Returns:
    ----------      
    Tensor or SparseTensor with dtype. If dtype is `None`, 
    dtype will be one of the following:       
        1. `graphgallery.floatx()` if `x` is floating.
        2. `graphgallery.intx()` if `x` is integer.
        3. `graphgallery.boolx()` if `x` is boolean.
    """
    backend = gg.backend(backend)
    device = parse_device(device, backend)
    if backend == "tensorflow":
        return tf_tensor.astensor(x, dtype=dtype, device=device, escape=escape)
    else:
        return th_tensor.astensor(x, dtype=dtype, device=device, escape=escape)


_astensors_fn = MultiInputs(type_check=False)(astensor)


def astensors(*xs, dtype=None, device=None, backend=None, escape=None):
    """Convert input matrices to Tensor(s) or SparseTensor(s).

    Parameters:
    ----------
    xs: one or a list of python object(s)
    dtype: The type of Tensor `x`, if not specified,
        it will automatically use appropriate data type.
        See `graphgallery.infer_type`.
    device: tf.device, optional. the desired device of returned tensor.
        Default: if `None`, uses the CPU device for the default tensor type.     
    backend: String or 'BackendModule', optional.
         `'tensorflow'`, `'torch'`, TensorFlowBackend, PyTorchBackend, etc.
         if not specified, return the current default backend module.    
    escape: a Class or a tuple of Classes,  `astensor` will disabled if
         `isinstance(x, escape)`.
    Returns:
    ----------      
    Tensor(s) or SparseTensor(s) with dtype. If dtype is `None`, 
    dtype will be one of the following:       
        1. `graphgallery.floatx()` if `x` is floating.
        2. `graphgallery.intx()` if `x` is integer.
        3. `graphgallery.boolx()` if `x` is boolean.
    """
    backend = gg.backend(backend)
    device = parse_device(device, backend)
    # escape
    return _astensors_fn(*xs, dtype=dtype, device=device, backend=backend, escape=escape)


def tensor2tensor(tensor, *, device=None):
    """Convert a TensorFLow tensor to PyTorch Tensor, or vice versa.
    """
    if gg.is_tensor(tensor, backend="tensorflow"):
        m = tensoras(tensor)
        device = parse_device(device, backend="torch")
        return astensor(m, device=device, backend="torch")
    elif gg.is_tensor(tensor, backend="torch"):
        m = tensoras(tensor)
        device = parse_device(device, backend="tensorflow")
        return astensor(m, device=device, backend="tensorflow")
    else:
        raise ValueError(f"The input must be a TensorFlow Tensor or PyTorch Tensor, buf got {type(tensor)}")


def tensoras(tensor):
    if gg.is_strided_tensor(tensor, backend="tensorflow"):
        m = tensor.numpy()
    elif gg.is_sparse(tensor, backend="tensorflow"):
        m = sparse_tensor_to_sparse_adj(tensor, backend="tensorflow")
    elif gg.is_strided_tensor(tensor, backend="torch"):
        m = tensor.detach().cpu().numpy()
        if m.ndim == 0:
            m = m.item()
    elif gg.is_sparse(tensor, backend="torch"):
        m = sparse_tensor_to_sparse_adj(tensor, backend="torch")
    elif isinstance(tensor, np.ndarray) or sp.isspmatrix(tensor):
        m = tensor.copy()
    else:
        m = np.asarray(tensor)
    return m


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
        return tf_tensor.sparse_adj_to_sparse_tensor(x)
    else:
        return th_tensor.sparse_adj_to_sparse_tensor(x)


def sparse_tensor_to_sparse_adj(x, *, backend=None):
    """Converts a SparseTensor to a Scipy sparse matrix (CSR matrix)."""
    backend = gg.backend(backend)

    if backend == "tensorflow":
        return tf_tensor.sparse_tensor_to_sparse_adj(x)
    else:
        return th_tensor.sparse_tensor_to_sparse_adj(x)


def sparse_edge_to_sparse_tensor(edge_index: np.ndarray, edge_weight: np.ndarray = None, shape: tuple = None, backend=None):

    backend = gg.backend(backend)
    if backend == "tensorflow":
        return tf_tensor.sparse_edge_to_sparse_tensor(edge_index, edge_weight, shape)
    else:
        return th_tensor.sparse_edge_to_sparse_tensor(edge_index, edge_weight, shape)


#### only works for tensorflow backend now #####################################

def normalize_adj_tensor(adj, rate=-0.5, fill_weight=1.0, backend=None):
    backend = gg.backend(backend)
    if backend == "tensorflow":
        return tf_tensor.normalize_adj_tensor(adj, rate=rate, fill_weight=fill_weight)
    else:
        # TODO
        return th_tensor.normalize_adj_tensor(adj, rate=rate, fill_weight=fill_weight)


def add_selfloops_edge(edge_index, edge_weight, n_nodes=None, fill_weight=1.0, backend=None):
    backend = gg.backend(backend)
    if backend == "tensorflow":
        return tf_tensor.normalize_adj_tensor(edge_index, edge_weight, n_nodes=n_nodes, fill_weight=fill_weight)
    else:
        # TODO
        return th_tensor.normalize_adj_tensor(edge_index, edge_weight, n_nodes=n_nodes, fill_weight=fill_weight)


def normalize_edge_tensor(edge_index, edge_weight=None, n_nodes=None, fill_weight=1.0, rate=-0.5, backend=None):
    backend = gg.backend(backend)
    if backend == "tensorflow":
        return tf_tensor.normalize_adj_tensor(edge_index, edge_weight=edge_weight, n_nodes=n_nodes, fill_weight=fill_weight, rate=rate)
    else:
        # TODO
        return th_tensor.normalize_adj_tensor(edge_index, edge_weight=edge_weight, n_nodes=n_nodes, fill_weight=fill_weight, rate=rate)
