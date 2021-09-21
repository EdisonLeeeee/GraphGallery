from .attacker import Attacker
from .flip_attacker import FlipAttacker

from . import targeted
from . import untargeted
from . import backdoor
from . import utils


def enabled_models(with_common=True):
    """Return the models in the gallery enabled by the current backend.

    Returns
    -------
    tuple
        A list of models enabled by the current backend.
    """
    return targeted.enabled_models(with_common) + untargeted.enabled_models(with_common)


attackers = enabled_models
