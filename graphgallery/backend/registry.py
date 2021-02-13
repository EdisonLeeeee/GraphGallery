import sys
import importlib
from graphgallery.backend import backend

__all__ = ["get_registry", "load_models", "load_models_only_tf_and_torch"]


def _gen_missing_model(model, backend):
    def _missing_model(*args, **kwargs):
        raise ImportError(f"model {model} is not supported by '{backend}'."
                          " You can switch to other backends by setting"
                          " the 'graphgallery.set_backend()'.")

    return _missing_model


def get_registry(mapping, backend_name=None):
    _backend = backend(backend_name)
    registry = mapping.get(_backend.abbr, None)
    if registry is None:
        raise RuntimeError(f"Currently {_backend} is not supported for this module.")
    return registry


def load_models(package, mapping, backend_name=None, sub_module=None):
    _backend = backend(backend_name)
    thismod = sys.modules[package]
    if sub_module:
        module_path = f".{sub_module}.{_backend.abbr}"
    else:
        module_path = f".{_backend.abbr}"
    importlib.import_module(module_path, package)

    for model, model_class in get_registry(mapping, _backend).items():
        setattr(thismod, model, model_class)

def load_models_only_tf_and_torch(package, mapping, backend_name=None, sub_module=None):
    _backend = backend(backend_name)
    thismod = sys.modules[package]
    if _backend == "tensorflow":
        _backend = backend("tensorflow")
    else:
        _backend = backend("pytorch")
        
    if sub_module:
        module_path = f".{sub_module}.{_backend.abbr}"
    else:
        module_path = f".{_backend.abbr}"
    importlib.import_module(module_path, package)

    for model, model_class in get_registry(mapping, _backend).items():
        setattr(thismod, model, model_class)