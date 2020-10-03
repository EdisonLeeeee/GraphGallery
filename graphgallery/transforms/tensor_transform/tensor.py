import torch
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from graphgallery import backend, is_sparse_tensor
from graphgallery.utils.device import parse_device
from graphgallery import transforms as T

__all__ = ["astensor", "astensors", "tensoras", "tensor2tensor",
           "sparse_adj_to_sparse_tensor",
           "sparse_tensor_to_sparse_adj",
           "sparse_edges_to_sparse_tensor",
           "normalize_adj_tensor", 
           "add_selfloops_edge", 
           "normalize_edge_tensor"]


def astensor(x, dtype=None, device=None, kind=None):
    """Convert input matrices to Tensor or SparseTensor.

    Parameters:
    ----------
    x: tf.Tensor, tf.Variable, Scipy sparse matrix, 
        Numpy array-like, etc.

    dtype: The type of Tensor `x`, if not specified,
        it will automatically using appropriate data type.
        See `graphgallery.infer_type`.

    device (:class:`torch.device` or `tf.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        
    kind: str
        "T" tf
        "P" torch

    Returns:
    ----------      
        Tensor or SparseTensor with dtype:       
        1. `graphgallery.floatx()` if `x` is floating
        2. `graphgallery.intx() ` if `x` is integer
        3. `Bool` if `x` is bool.
    """
    if kind is None:
        kind = backend().kind
    else:
        assert kind in {"T", "P"}
    device = parse_device(device, kind)
    if kind == "T":
        return T.tf_tensor.astensor(x, dtype=dtype, device=device)
    else:
        return T.th_tensor.astensor(x, dtype=dtype, device=device)


def astensors(*xs, device=None, kind=None):
    """Convert input matrices to Tensor(s) or SparseTensor(s).

    Parameters:
    ----------
    xs: tf.Tensor, tf.Variable, Scipy sparse matrix, 
        Numpy array-like, or a list of them, etc.

    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
        
    kind: str
        "T" tf
        "P" torch        

    Returns:
    ----------      
        Tensor(s) or SparseTensor(s) with dtype:       
        1. `graphgallery.floatx()` if `x` in `xs` is floating
        2. `graphgallery.intx() ` if `x` in `xs` is integer
        3. `Bool` if `x` in `xs` is bool.
    """
    if kind is None:
        kind = backend().kind
    else:
        assert kind in {"T", "P"}
    if kind == "T":
        return T.tf_tensor.astensors(*xs, device=device)
    else:
        return T.th_tensor.astensors(*xs, device=device)
    
    
def tensor2tensor(tensor, device=None):
    """Convert a TensorFLow tensor to PyTorch Tensor,
    or vice versa
    """
    if is_tf_tensor(tensor):
        m = T.tensoras(tensor)
        device = parse_device(device, "P")
        return T.th_tensor.astensor(m, device=device)
    elif is_th_tensor(tensor):
        m = T.tensoras(tensor)
        device = parse_device(device, "T")
        return T.tf_tensor.astensor(m, device=device)
    else:
        raise ValueError(f"The input must be a Tensorflow Tensor or PyTorch Tensor, buf got {type(tensor)}")
        
def tensoras(tensor):
    if isinstance(tensor, (tf.Tensor, tf.Variable)):
        m = tensor.numpy()
    elif is_sparse_tensor(tensor):
        m = T.sparse_tensor_to_sparse_adj(tensor)
    elif isinstance(tensor, torch.Tensor):
        m = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray) or sp.isspmatrix(tensor):
        m = tensor.copy()
    else:
        m = np.asarray(tensor)
    return m

def sparse_adj_to_sparse_tensor(x, kind=None):
    """Converts a Scipy sparse matrix to a TensorFlow/PyTorch SparseTensor.

    Parameters
    ----------
        x: scipy.sparse.sparse
            Matrix in Scipy sparse format.
    Returns
    -------
        S: SparseTensor
            Matrix as a sparse tensor.
    """
    if kind is None:
        kind = backend().kind
    else:
        assert kind in {"T", "P"}
        
    if kind == "T":
        return T.tf_tensor.sparse_adj_to_sparse_tensor(x)
    else:
        return T.th_tensor.sparse_adj_to_sparse_tensor(x)

def sparse_tensor_to_sparse_adj(x, kind=None):
    """Converts a SparseTensor to a Scipy sparse matrix (CSR matrix)."""
    if kind is None:
        kind = backend().kind
    else:
        assert kind in {"T", "P"}
        
    if kind == "T":
        return T.tf_tensor.sparse_tensor_to_sparse_adj(x)
    else:
        return T.th_tensor.sparse_tensor_to_sparse_adj(x)

def sparse_edges_to_sparse_tensor(edge_index: np.ndarray, edge_weight: np.ndarray = None, shape: tuple = None, kind=None):

    if kind is None:
        kind = backend().kind
    else:
        assert kind in {"T", "P"}
    if kind == "T":
        return T.tf_tensor.sparse_edges_to_sparse_tensor(edge_index, edge_weight, shape)
    else:
        return T.th_tensor.sparse_edges_to_sparse_tensor(edge_index, edge_weight, shape)


#### only work for tensorflow backend #####################################

def normalize_adj_tensor(adj, rate=-0.5, fill_weight=1.0, kind=None):
    if kind is None:
        kind = backend().kind
    else:
        assert kind in {"T", "P"}
    if kind == "T":
        return T.tf_tensor.normalize_adj_tensor(adj, rate=rate, fill_weight=fill_weight)
    else:
        return T.th_tensor.normalize_adj_tensor(adj, rate=rate, fill_weight=fill_weight)


def add_selfloops_edge(edge_index, edge_weight, n_nodes=None, fill_weight=1.0, kind=None):
    if kind is None:
        kind = backend().kind
    else:
        assert kind in {"T", "P"}
    if kind == "T":
        return T.tf_tensor.normalize_adj_tensor(edge_index, edge_weight, n_nodes=n_nodes, fill_weight=fill_weight)
    else:
        return T.th_tensor.normalize_adj_tensor(edge_index, edge_weight, n_nodes=n_nodes, fill_weight=fill_weight)


def normalize_edge_tensor(edge_index, edge_weight=None, n_nodes=None, fill_weight=1.0, rate=-0.5, kind=None):
    if kind is None:
        kind = backend().kind
    else:
        assert kind in {"T", "P"}
    if kind == "T":
        return T.tf_tensor.normalize_adj_tensor(edge_index, edge_weight=edge_weight, n_nodes=n_nodes, fill_weight=fill_weight, rate=rate)
    else:
        return T.th_tensor.normalize_adj_tensor(edge_index, edge_weight=edge_weight, n_nodes=n_nodes, fill_weight=fill_weight, rate=rate)
