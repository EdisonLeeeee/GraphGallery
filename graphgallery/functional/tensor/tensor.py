import torch
import numpy as np
import scipy.sparse as sp
from typing import Any, Optional

import graphgallery as gg
from graphgallery import functional as gf
from graphgallery.typing import Device, Backend

from . import tensorflow
from . import pytorch
from .ops import get_module


def data_type_dict(backend: Optional[Backend] = None) -> dict:
    module = get_module(backend)
    return module.data_type_dict()


def is_sparse(x: Any, backend: Optional[Backend] = None) -> bool:
    """Check whether 'x' is a sparse Tensor.

    Parameters:
    ----------
    x: A python object to check.

    backend: String or 'BackendModule', optional.    
     ''tensorflow'', ''torch'', TensorFlowBackend, PyTorchBackend, etc.    
     if not specified, return the current default backend module. 
    Returns:
    ----------
    'True' iff 'x' is a (tf or torch) sparse-tensor.
    """
    module = get_module(backend)
    return module.is_sparse(x)


def is_dense(x: Any, backend: Optional[Backend] = None) -> bool:
    """Check whether 'x' is a strided (dense) Tensor.

    Parameters:
    ----------
    x: A python object to check.

    backend: String or 'BackendModule', optional.    
     ''tensorflow'', ''torch'', TensorFlowBackend, PyTorchBackend, etc.    
     if not specified, return the current default backend module. 

    Returns:
    ----------
    'True' iff 'x' is a (tf or torch) strided (dense) Tensor.
    """

    module = get_module(backend)
    return module.is_dense(x)


def is_tensor(x: Any, backend: Optional[Backend] = None) -> bool:
    """Check whether 'x' is 
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
     ''tensorflow'', ''torch'', TensorFlowBackend, PyTorchBackend, etc.    
     if not specified, return the current default backend module.    
    Returns:
    ----------
    'True' iff 'x' is a (tf or torch) (sparse-)tensor.
    """
    module = get_module(backend)
    return module.is_tensor(x)


def astensor(x, *, dtype=None, device: Optional[Device] = None,
             backend: Optional[Backend] = None, escape=None):
    """Convert input object to Tensor or SparseTensor.

    Parameters:
    ----------
    x: any python object

    dtype: The type of Tensor 'x', if not specified,
        it will automatically use appropriate data type.
        See 'graphgallery.infer_type'.
    device: tf.device, optional. the desired device of returned tensor.
        Default: if 'None', uses the CPU device for the default tensor type.
    backend: String or 'BackendModule', optional.
         ''tensorflow'', ''torch'', TensorFlowBackend, PyTorchBackend, etc.
         if not specified, return the current default backend module. 
    escape: a Class or a tuple of Classes,  'astensor' will disabled if
         'isinstance(x, escape)'.

    Returns:
    ----------      
    Tensor or SparseTensor with dtype. If dtype is 'None', 
    dtype will be one of the following:       
        1. 'graphgallery.floatx()' if 'x' is floating.
        2. 'graphgallery.intx()' if 'x' is integer.
        3. 'graphgallery.boolx()' if 'x' is boolean.
    """

    backend = gg.backend(backend)
    device = gf.device(device, backend)
    module = get_module(backend)
    return module.astensor(x, dtype=dtype,
                           device=device,
                           escape=escape)


_astensors_fn = gf.multiple(type_check=False)(astensor)


def astensors(*xs, dtype=None, device: Optional[Device] = None,
              backend: Optional[Backend] = None, escape=None):
    """Convert input matrices to Tensor(s) or SparseTensor(s).

    Parameters:
    ----------
    xs: one or a list of python object(s)
    dtype: The type of Tensor 'x', if not specified,
        it will automatically use appropriate data type.
        See 'graphgallery.infer_type'.
    device: tf.device, optional. the desired device of returned tensor.
        Default: if 'None', uses the CPU device for the default tensor type.     
    backend: String or 'BackendModule', optional.
         'tensorflow', 'torch', TensorFlowBackend, PyTorchBackend, etc.
         if not specified, return the current default backend module.    
    escape: a Class or a tuple of Classes,  `astensor` will disabled if
         `isinstance(x, escape)`.

    Returns:
    -------     
    Tensor(s) or SparseTensor(s) with dtype. If dtype is 'None', 
    dtype will be one of the following:       
        1. 'graphgallery.floatx()' if 'x' is floating.
        2. 'graphgallery.intx()' if 'x' is integer.
        3. 'graphgallery.boolx()' if 'x' is boolean.
    """
    backend = gg.backend(backend)
    device = gf.device(device, backend)
    return _astensors_fn(*xs,
                         dtype=dtype,
                         device=device,
                         backend=backend,
                         escape=escape)


def tensor2tensor(tensor, *, device: Optional[Device] = None):
    """Convert a TensorFLow tensor to PyTorch Tensor, or vice versa.
    """
    if tensorflow.is_tensor(tensor):
        m = tensoras(tensor)
        device = gf.device(device, backend="torch")
        return astensor(m, device=device, backend="torch")
    elif pytorch.is_tensor(tensor):
        m = tensoras(tensor)
        device = gf.device(device, backend="tensorflow")
        return astensor(m, device=device, backend="tensorflow")
    else:
        raise ValueError(
            f"The input must be a TensorFlow or PyTorch Tensor, buf got {type(tensor).__name__}"
        )


def tensoras(tensor):
    if tensorflow.is_dense(tensor):
        m = tensor.numpy()
    elif tensorflow.is_sparse(tensor):
        m = tensorflow.sparse_tensor_to_sparse_adj(tensor)
    elif pytorch.is_dense(tensor):
        m = tensor.detach().cpu().numpy()
        if m.ndim == 0:
            m = m.item()
    elif pytorch.is_sparse(tensor):
        m = pytorch.sparse_tensor_to_sparse_adj(tensor)
    elif isinstance(tensor, np.ndarray) or sp.isspmatrix(tensor):
        m = tensor.copy()
    else:
        m = np.asarray(tensor)
    return m
