"""Inspired by Keras backend config API. https://tensorflow.google.com """

import importlib
import sys

from .modules import BackendModule, TensorFlowBackend, PyTorchBackend, PyGBackend, DGLBackend

__all__ = [
    'allowed_backends', 'backend_dict', 'backend', 'set_backend',
    'set_to_default_backend', 'boolx', 'intx', 'set_intx', 'floatx',
    'set_floatx', 'epsilon', 'set_epsilon', 'file_ext', 'set_file_ext'
]

# used to store the models or weights for `TensorFlow` and `PyTorch`
_EXT = ".h5"

##### Backends ######
_TF = 'tensorflow'
_TORCH = 'torch'

_DEFAULT_BACKEND = PyTorchBackend()
_BACKEND = _DEFAULT_BACKEND

_ALL_BACKENDS = {
    TensorFlowBackend, PyTorchBackend, PyGBackend, DGLBackend,
}
_BACKEND_DICT = {}


def allowed_backends():
    """Return the allowed backends."""
    return tuple(backend_dict().keys())


def backend_dict():
    return _BACKEND_DICT


def set_backend_dict():
    global _BACKEND_DICT
    _BACKEND_DICT = {}
    for bkd in _ALL_BACKENDS:
        for name in bkd.alias:
            _BACKEND_DICT[name] = bkd


##### Types ######
_INT_TYPES = {'uint8', 'int8', 'int16', 'int32', 'int64'}
_FLOAT_TYPES = {'float16', 'float32', 'float64'}

# The type of integer to use throughout a network
_INTX = 'int64'
# The type of float to use throughout a network
_FLOATX = 'float32'
# The type of bool to use throughout a network
_BOOLX = 'bool'

_EPSILON = 1e-7


def epsilon():
    """Returns the value of the fuzz factor used in numeric expressions.

    Returns:
        A float.

    Example:
    >>> graphgallery.epsilon()
    1e-07
    """
    return _EPSILON


def set_epsilon(value):
    """Sets the value of the fuzz factor used in numeric expressions.

    Args:
        value: float. New value of epsilon.

    Example:
    >>> graphgallery.epsilon()
    1e-07
    >>> graphgallery.set_epsilon(1e-5)
    >>> graphgallery.epsilon()
    1e-05
     >>> graphgallery.set_epsilon(1e-7)
    """
    global _EPSILON
    _EPSILON = value


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
        raise ValueError(
            f"Unknown floatx type: '{str(dtype)}', expected one of {_FLOAT_TYPES}."
        )
    global _FLOATX
    _FLOATX = str(dtype)
    # torch.set_default_tensor_type(torch.HalfTensor)
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
        raise ValueError(
            f"Unknown integer type: '{str(dtype)}', expected one of {_INT_TYPES}."
        )
    global _INTX

    if _BACKEND == _TORCH and dtype != 'int64':
        raise RuntimeError(
            f"For {_BACKEND}, tensors used as integer must be 'long' ('int64'), not '{str(dtype)}'."
        )

    _INTX = str(dtype)
    return _INTX


def backend(module_name=None):
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
    >>> graphgallery.backend('torch')
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
                f"Unsupported backend module name: '{module_name}', expected one of {allowed_backends()}."
            )
        return module()


def set_to_default_backend():
    """Set the current backend to default"""
    global _BACKEND
    _BACKEND = _DEFAULT_BACKEND
    # Using `int32` is more efficient
    set_intx('int32')
    return _BACKEND


def set_backend(module_name=None):
    """Set the default backend module.

    Parameters:
    ----------
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
            # gallery models
            from graphgallery.gallery import nodeclas
            from graphgallery.gallery import graphclas
            from graphgallery.gallery import linkpred
            importlib.reload(nodeclas)
            importlib.reload(graphclas)
            importlib.reload(linkpred)
            # attacker models
            from graphgallery.attack import targeted
            from graphgallery.attack import untargeted
            from graphgallery.attack import backdoor
            importlib.reload(targeted)
            importlib.reload(untargeted)
            importlib.reload(backdoor)
        except Exception as e:
            print(
                f"Something went wrong. Set to Default Backend {_DEFAULT_BACKEND}.",
                file=sys.stderr)
            set_to_default_backend()
            raise e

    return _BACKEND


def file_ext():
    """Returns the checkpoint filename suffix(extension) for the training model

    Returns
    -------
    str
        ".h5" by default
    """
    return _EXT


def set_file_ext(ext):
    """Set the filename suffix(extension)"""
    global _EXT
    _EXT = ext
    return _EXT


set_backend_dict()
