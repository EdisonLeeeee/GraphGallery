from graphgallery.utils.decorators import EqualVarLength, get_length
from graphgallery.utils.bunch import Bunch
from graphgallery.utils.context_manager import nullcontext
from graphgallery.utils.raise_error import raise_if_kwargs, assert_kind
from graphgallery.utils.device import parse_device
from graphgallery.utils.shape import repeat
from graphgallery.utils.tqdm import tqdm
from graphgallery.utils.degree import degree_mixing_matrix, degree_assortativity_coefficient
from graphgallery.utils.context_manager import nullcontext
from graphgallery.utils.ego import ego_graph
