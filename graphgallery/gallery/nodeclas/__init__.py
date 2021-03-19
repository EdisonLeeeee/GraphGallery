from .trainer import Trainer
from .registered_models import (TensorFlow, PyTorch, PyG,
                                DGL_PyTorch, DGL_TensorFlow,
                                Common,
                                MAPPING)


import graphgallery
from functools import partial


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
    return model in enabled_models()


def enabled_models(with_common=True):
    """Return the models in the gallery enabled by the current backend.

    Parameters
    ----------
    with_common : bool
        Whether to return common models (framework-agnostic).

    Returns
    -------
    graphgallry.functional.BuhcnDict
        A dict of models enabled by the current backend.
    """
    return get_registry() + Common


graphgallery.load_models(__name__, mapping=MAPPING)
get_registry = partial(graphgallery.get_registry, mapping=MAPPING)
models = enabled_models
