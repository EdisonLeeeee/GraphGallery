"""Inspired by Keras backend config API. https://tensorflow.google.com """

import importlib
import sys
from tensorflow.keras import backend as K
from typing import Union, Tuple, Optional

from .modules import BackendModule, TensorFlowBackend, PyTorchBackend, PyGBackend, DGLPyTorchBackend, DGLTensorFlowBackend

__all__ = ['allowed_backends', 'backend_dict',
           'backend', 'set_backend', 'set_to_default_backend',
           'boolx', 'intx', 'set_intx',
           'floatx', 'set_floatx',
           'epsilon', 'set_epsilon',
           'file_postfix', 'set_file_postfix']


# used to store the models or weights for `TensorFlow` and `PyTorch`
_POSTFIX = ".h5"

##### Backends ######
_TF = 'tensorflow'
_TORCH = 'torch'
_MXNET = 'mxnet'
_CUPY = 'cupy'
_NUMPY = 'numpy'

_DEFAULT_BACKEND = TensorFlowBackend()
_BACKEND = _DEFAULT_BACKEND

_ALL_BACKENDS = {TensorFlowBackend, PyTorchBackend, PyGBackend, DGLPyTorchBackend, DGLTensorFlowBackend}
_BACKEND_DICT = {}

BACKEND_TYPE = Union[TensorFlowBackend, PyTorchBackend, PyGBackend, DGLPyTorchBackend, DGLTensorFlowBackend]


def allowed_backends() -> Tuple[str]:
    """
    Return tuple of tuple of tuple of tuple of tuple of tuple ( tuple tuple.

    Args:
    """
    return tuple(backend_dict().keys())


def backend_dict() -> dict:
    """
    Returns a dictionary of backend_dict.

    Args:
    """
    return _BACKEND_DICT


def set_backend_dict():
    """
    Sets the global key / values.

    Args:
    """
    global _BACKEND_DICT
    _BACKEND_DICT = {}
    for bkd in _ALL_BACKENDS:
        for act in bkd.alias:
            _BACKEND_DICT[act] = bkd


##### Types ######
_INT_TYPES = {'uint8', 'int8', 'int16', 'int32', 'int64'}
_FLOAT_TYPES = {'float16', 'float32', 'float64'}

# The type of integer to use throughout a network
_INTX = 'int32'
# The type of float to use throughout a network
_FLOATX = 'float32'
# The type of bool to use throughout a network
_BOOLX = 'bool'

epsilon = K.epsilon
set_epsilon = K.set_epsilon


def boolx() -> str:
    """Returns the default bool type, as a string,
        i.e., bool

    Returns:
    --------
    String, the current default bool type.

    Example:
    --------
    >>> graphgallery.boolx()
    'bool'
    """
    return _BOOLX


def floatx() -> str:
    """Returns the default float type, as a string.

    E.g. `'float16'`, `'float32'`, `'float64'`.

    Returns:
    --------
    String, the current default float type.

    Example:
    --------
    >>> graphgallery.floatx()
    'float32'
    """
    return _FLOATX


def set_floatx(dtype: str) -> str:
    """Sets the default float type.

    Parameters:
    --------
    dtype: String; `'float16'`, `'float32'`, or `'float64'`.

    Example:
    --------
    >>> graphgallery.floatx()
    'float32'
    >>> graphgallery.set_floatx('float64')
    'float64'

    Raises:
    --------
    ValueError: In case of invalid value.
    """

    if dtype not in _FLOAT_TYPES:
        raise ValueError(f"Unknown floatx type: '{str(dtype)}', expected one of {_FLOAT_TYPES}.")
    global _FLOATX
    _FLOATX = str(dtype)
    return _FLOATX


def intx() -> str:
    """Returns the default integer type, as a string.

    E.g. `'uint8'`, `'int8'`, `'int16'`, 
        `'int32'`, `'int64'`.

    Returns:
    --------
    String, the current default integer type.

    Example:
    --------
    >>> graphgallery.intx()
    'int32'

    Note:
    -------
    The default integer type of PyTorch backend will set to
        'int64', i.e., 'Long'.
    """
    return _INTX


def set_intx(dtype: str) -> str:
    """Sets the default integer type.

    Parameters:
    --------
    dtype: String. `'uint8'`, `'int8'`, `'int16'`, 
        `'int32'`, `'int64'`.

    Example:
    --------
    >>> graphgallery.intx()
    'int32'
    >>> graphgallery.set_intx('int64')
    'int64'

    Raises:
    --------
    ValueError: In case of invalid value.
    RuntimeError: PyTorch backend using other integer types except for 'int64.
    """

    if dtype not in _INT_TYPES:
        raise ValueError(f"Unknown integer type: '{str(dtype)}', expected one of {_INT_TYPES}.")
    global _INTX

    if _BACKEND == _TORCH and dtype != 'int64':
        raise RuntimeError(
            f"For {_BACKEND}, tensors used as integer must be 'long' ('int64'), not '{str(dtype)}'.")

    _INTX = str(dtype)
    return _INTX


def backend(module_name: Optional[Union[str, BackendModule]] = None) -> BACKEND_TYPE:
    """Publicly accessible method
    for determining the current backend.

    Parameters:
    --------
    module_name: String or 'BackendModule', optional.
     `'tensorflow'`, `'torch'`, TensorFlowBackend, PyTorchBackend, etc.
     if not specified, return the current default backend module. 

    Returns:
    --------
    The backend module.

    E.g. `'TensorFlow 2.1.2 Backend'`,
      `'PyTorch 1.6.0+cpu Backend'`.

    Example:
    --------
    >>> graphgallery.backend()
    'TensorFlow 2.1.2 Backend'
    >>> graphgallery.backend('torch)
    'PyTorch 1.6.0+cpu Backend'    
    """
    if module_name is None:
        return _BACKEND
    elif isinstance(module_name, BackendModule):
        return module_name
    else:
        module_name = str(module_name)
        module = _BACKEND_DICT.get(module_name.lower(), None)
        
        if module is None:
            raise ValueError(
                f"Unsupported backend module name: '{module_name}', expected one of {allowed_backends()}.")
        return module()

def set_to_default_backend():
    """
    Sets the default backend.

    Args:
    """
    global _BACKEND
    _BACKEND = _DEFAULT_BACKEND
    # Using `int32` is more efficient
    set_intx('int32')    
    return _BACKEND
    
def set_backend(module_name: Optional[Union[str, BackendModule]] = None) -> BACKEND_TYPE:
    """Set the default backend module.

    Parameters:
    --------
    module_name: String or 'BackendModule', optional.
        `'tf'`, `'tensorflow'`,
        `'th'`, `'torch'`, `'pytorch'`.

    Example:
    --------
    >>> graphgallery.backend()
    'TensorFlow 2.1.2 Backend'

    >>> graphgallery.set_backend('torch')
    'PyTorch 1.6.0+cpu Backend'

    Raises:
    --------
    ValueError: In case of invalid value.
    """

    _backend = backend(module_name)


    global _BACKEND

    if _backend != _BACKEND:
        _BACKEND = _backend
        if _backend == _TORCH:
            # PyTorch backend uses `int64` as default
            set_intx('int64')
        else:
            # Using `int32` is more efficient
            set_intx('int32')
        try:
            from graphgallery.nn import gallery
            importlib.reload(gallery)
        except Exception as e:
            print(f"Something went wrong. Set to Default Backend {_DEFAULT_BACKEND}.", file=sys.stderr)
            set_to_default_backend()
            raise e        
        
    return _BACKEND


def file_postfix():
    """
    Return postfix postfix.

    Args:
    """
    return _POSTFIX


def set_file_postfix(postfix):
    """
    Sets the postfix.

    Args:
        postfix: (str): write your description
    """
    global _POSTFIX
    _POSTFIX = postfix
    return _POSTFIX


set_backend_dict()
