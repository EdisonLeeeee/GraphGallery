from .targeted_attacker import TargetedAttacker
from .registered_models import TensorFlow, PyTorch, Common, MAPPING
from .common import *


import graphgallery
from functools import partial


def is_enabled(attacker: str) -> bool:
    """Return true if the attacker is enabled by the current backend.

    Parameters
    ----------
    attacker : str
        The attacker name.

    Returns
    -------
    bool
        True if the attacker is enabled by the current backend.
    """
    return attacker in enabled_models()


def enabled_models(with_common=True):
    """Return the models in the gallery enabled by the current backend.

    Returns
    -------
    tuple
        A list of models enabled by the current backend.
    """
    return get_registry() + Common


graphgallery.load_models_only_tf_and_torch(__name__, mapping=MAPPING, sub_module=None)
get_registry = partial(graphgallery.get_registry, mapping=MAPPING)
attackers = enabled_models
