from .torch_keras import TorchKeras
# from .tf_keras import TFKeras

__all__ = ["TorchKeras", "get_model"]

import importlib
import graphgallery as gg


def get_model(model: str, backend_name=None):
    backend = gg.backend(backend_name)
    name = model.split('.')
    assert len(name) <= 2
    if len(name) >= 2:
        sub_module, model = name
        mod = importlib.import_module(f".{backend.abbr}.{sub_module}", __name__)
        _model_class = mod.__dict__.get(model, None)

    else:
        mod = importlib.import_module(f".{backend.abbr}", __name__)
        _model_class = mod.__dict__.get(model, None)

    if _model_class is not None:
        return _model_class
    else:
        raise ImportError(f"model {model} is not supported by '{backend}'."
                          " You can switch to other backends by setting"
                          " the 'graphgallery.backend' environment.")
