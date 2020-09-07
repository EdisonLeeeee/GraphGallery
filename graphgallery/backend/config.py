"""Inspired by Keras backend config API. https://tensorflow.google.com """

import importlib
import graphgallery
from graphgallery.backend.modules import TensorFlowBackend, PyTorchBackend
from tensorflow.keras import backend as K

_BACKEND = TensorFlowBackend()
_MODULE_NAME = {"tf": "T", "tensorflow": "T", "T": "T",
                "th": "P", "torch": "P", "pytorch": "P", "P": "P"}

_ALLOWED_NAME = set(list(_MODULE_NAME.keys()))

_MODULE = {"T": TensorFlowBackend,
           "P": PyTorchBackend}


# The type of integer to use throughout a session.
_INTX = 'int32'
# The type of float to use throughout a session.
_FLOATX = 'float32'

epsilon = K.epsilon
set_epsilon = K.set_epsilon


def floatx():
    """Returns the default float type, as a string.

    E.g. `'float16'`, `'float32'`, `'float64'`.

    Returns:
      String, the current default float type.

    Example:
    >>> graphgallery.floatx()
    'float32'
    """
    return _FLOATX


def set_floatx(dtype):
    """Sets the default float type.

    Parameters:
      dtype: String; `'float16'`, `'float32'`, or `'float64'`.

    Example:
    >>> graphgallery.floatx()
    'float32'
    >>> graphgallery.set_floatx('float64')
    >>> graphgallery.floatx()
    'float64'

    Raises:
      ValueError: In case of invalid value.
    """

    if dtype not in {'float16', 'float32', 'float64'}:
        raise ValueError('Unknown floatx type: ' + str(dtype))
    global _FLOATX
    _FLOATX = str(dtype)
    return _FLOATX


def intx():
    """Returns the default integer type, as a string.

    E.g. `'int16'`, `'int32'`, `'int64'`.

    Returns:
      String, the current default integer type.

    Example:
    >>> graphgallery.intx()
    'int32'
    """
    return _INTX


def set_intx(dtype):
    """Sets the default integer type.

    Parameters:
      dtype: String; `'int16'`, `'int32'`, or `'int64'`.

    Example:
    >>> graphgallery.intx()
    'int32'
    >>> graphgallery.set_intx('int64')
    >>> graphgallery.intx()
    'int64'

    Raises:
      ValueError: In case of invalid value.
    """

    if dtype not in {'int16', 'int32', 'int64'}:
        raise ValueError('Unknown floatx type: ' + str(dtype))
    global _INTX
    _INTX = str(dtype)
    return _INTX


def backend():
    """Returns the default backend module, as a string.

    E.g. `'TensorFlow 2.1.0 Backend'`,
      `'PyTorch 1.6.0+cpu Backend'`.

    Returns:
      String, the current default backend module.

    Example:
    >>> graphgallery.backend()
    'TensorFlow 2.1.0 Backend'
    """
    return _BACKEND


def set_backend(module_name):
    """Sets the default backend module.

    Parameters:
      module_name: String; `'tf'`, `'tensorflow'`,
             `'th'`, `'torch'`, 'pytorch'`.

    Example:
    >>> graphgallery.backend()
    'TensorFlow 2.1.0 Backend'
    >>> graphgallery.set_backend('torch')
    >>> graphgallery.backend()
    'PyTorch 1.6.0+cpu Backend'

    Raises:
      ValueError: In case of invalid value.
    """
    if module_name not in _ALLOWED_NAME:
        raise ValueError(
            f"Unknown module name: '{str(module_name)}', expected one of {_ALLOWED_NAME}.")
    global _BACKEND
    name = _MODULE_NAME.get(module_name)
    if name != _BACKEND.kind:
        # TODO: improve
        _BACKEND = _MODULE.get(name)()
        importlib.reload(graphgallery)
        importlib.reload(graphgallery.nn)
        importlib.reload(graphgallery.nn.models)

    return _BACKEND
