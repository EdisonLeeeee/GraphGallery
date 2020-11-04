from .model import Model
from .gallery_model import GalleryModel
from .semisupervised.semi_supervised_model import SemiSupervisedModel
from .unsupervised.unsupervised_model import UnsupervisedModel

import sys
import importlib
from graphgallery import backend
from typing import Tuple

__all__ = ['Model']

_SEMI_SUPERVISED_MODELS = {"GCN",
           "GAT",
           "ClusterGCN",
           "SGC",
           "GWNN",
           "RobustGCN",
           "GraphSAGE",
           "FastGCN",
           "ChebyNet",
           "DenseGCN",
           "LGCN",
           "OBVAT",
           "SBVAT",
           "GMNN",
           "DAGNN",
            # Experimental models                           
           "EdgeGCN",
           "SimplifiedOBVAT",
           "GCN_MIX",
           "GCNA",
           "SAT"
}

_UNSUPERVISED_MODELS = {
           "Node2vec",
           "Deepwalk",
}

_enabled_models = set()



def _gen_missing_model(model, backend):
    def _missing_model(*args, **kwargs):
        raise ImportError(f"model {model} is not supported by '{backend}'."
                          " You can switch to other backends by setting"
                          " the 'graphgallery.backend' environment.")
    return _missing_model

def load_models(backend_name=None):
    _backend = backend(backend_name)
    mod = importlib.import_module(f".semisupervised.{_backend.abbr}", __name__)
    thismod = sys.modules[__name__]
    
    for model in _SEMI_SUPERVISED_MODELS:
        _model_class = mod.__dict__.get(model, None)
        
        if _model_class is not None:
            _enabled_models.add(model)
            setattr(thismod, model, _model_class)
        else:
            setattr(thismod, model, _gen_missing_model(model, _backend))
            
    mod = importlib.import_module(f".unsupervised", __name__)
            
    for model in _UNSUPERVISED_MODELS:
        _model_class = mod.__dict__.get(model, None)
        
        if _model_class is not None:
            _enabled_models.add(model)
            setattr(thismod, model, _model_class)
        else:
            setattr(thismod, model, _gen_missing_model(model, _backend))            


def is_enabled(model: str) -> bool:
    """Return true if the model is enabled by the current backend.

    Parameters
    ----------
    model : str
        The model name.

    Returns
    -------
    bool
        True if the model is enabled by the current backend.
    """
    return model in _enabled_models

def enabled_models() -> Tuple[str]:
    """Return the models in the gallery enabled by the current backend.

    Returns
    -------
    tuple
        A list of models enabled by the current backend.
    """    
    return tuple(_enabled_models)

load_models()