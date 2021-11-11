import importlib
import sys

from .modules import BackendModule, PyTorchBackend, PyGBackend, DGLBackend

__all__ = [
    'allowed_backends', 'backend_dict', 'backend', 'set_backend',
    'set_to_default_backend', 'file_ext', 'set_file_ext'
]

# used to store the models or weights for `PyTorch`
_EXT = ".pth"

_DEFAULT_BACKEND = PyTorchBackend()
_BACKEND = _DEFAULT_BACKEND

_ALL_BACKENDS = {PyTorchBackend, PyGBackend, DGLBackend, }
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


def backend(module_name=None):
    """Publicly accessible method
    for determining the current backend.

    Parameters:
    --------
    module_name: String or 'BackendModule', optional.
     `'torch'`, `PyTorchBackend`, `'pyg`, etc.
     if not specified, return the current default backend module. 

    Returns:
    --------
    The backend module.

    E.g. `'PyTorch 1.6.0+cpu Backend'`.

    Example:
    --------
    >>> graphgallery.backend()
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
    return _BACKEND


def set_backend(module_name=None):
    """Set the default backend module.

    Parameters:
    ----------
    module_name: String or 'BackendModule', optional.
        `'th'`, `'torch'`, `'pytorch'`.

    Example:
    --------
    >>> graphgallery.backend()
    'PyTorch 1.6.0+cpu Backend'

    Raises:
    --------
    ValueError: In case of invalid value.
    """

    _backend = backend(module_name)

    global _BACKEND

    if _backend != _BACKEND:
        _BACKEND = _backend
        try:
            # gallery models
            from graphgallery.gallery import nodeclas
            from graphgallery.gallery import graphclas
            from graphgallery.gallery import linkpred
            importlib.reload(nodeclas)
            importlib.reload(graphclas)
            importlib.reload(linkpred)
        except Exception as e:
            print(
                f"Something went wrong when switching to other backend.",
                file=sys.stderr)
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
