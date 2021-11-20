from .nodeclas_trainer import NodeClasTrainer
from .registered_models import (PyTorch, PyG, DGL, MAPPING)

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


def enabled_models():
    """Return the models in the gallery enabled by the current backend.

    Returns
    -------
    graphgallry.functional.BuhcnDict
        A dict of models enabled by the current backend.
    """
    return get_registry()


graphgallery.load_models(__name__, mapping=MAPPING)
get_registry = partial(graphgallery.get_registry, mapping=MAPPING)
models = enabled_models
