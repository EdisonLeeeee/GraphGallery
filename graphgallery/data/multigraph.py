import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from functools import lru_cache
from copy import copy as copy_fn

from graphgallery.data.base_graph import BaseGraph


class MultiGraph(BaseGraph):
    """Attributed labeled multigraph stored in a list of sparse matrix form."""
    ...
