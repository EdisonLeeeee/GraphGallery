# backend
from graphgallery.backend.gpu import set_memory_growth
from graphgallery.backend.config import (epsilon, floatx, intx, backend, POSTFIX,
                                         set_epsilon, set_floatx, set_intx, set_backend)


from graphgallery.utils.type_check import *




from graphgallery.utils.shape import repeat
from graphgallery.utils.tqdm import tqdm
from graphgallery.utils.degree import degree_mixing_matrix, degree_assortativity_coefficient
from graphgallery.utils.context_manager import nullcontext
from graphgallery.utils.ego import ego_graph
from graphgallery.utils.bunch import Bunch


# from graphgallery.utils.export import _NAME_TO_SYMBOL_MAPPING
# for _api_name, _api in _NAME_TO_SYMBOL_MAPPING.items():
#     exec(f"{_api_name}=_api")

# Base modules
from graphgallery import nn
from graphgallery import utils
from graphgallery import sequence
from graphgallery import data
from graphgallery import backend
from graphgallery import transforms


__version__ = '0.4.1'

__all__ = ['graphgallery', 'nn', 'utils', 'sequence', 'data',
           'backend', 'transforms', '__version__']
