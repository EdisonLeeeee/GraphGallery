import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from functools import lru_cache
from collections import Counter
from copy import copy as copy_fn

from typing import Union, Optional, List, Tuple, Any

from .base_graph import BaseGraph
from .preprocess import largest_connected_components, create_subgraph
from .utils import check


class MultiGraph(BaseGraph):
    """Attributed labeled multigraph stored in a list of sparse matrix form."""

    def __init__(self, adj_matrix=None,
                 node_attr=None,
                 node_labels=None,
                 edge_attr=None,
                 edge_labels=None,
                 graph_labels=None,
                 graph_attr=None,
                 metadata: Any = None,
                 copy: bool = True):
        r"""Create a multiple (un)dirtected (attributed and labeled) graph.

        """
        adj_matrix, node_attr, node_labels = check(
            adj_matrix, node_attr, node_labels, copy=copy)

        local_vars = locals()
        del local_vars['self'], local_vars['copy']
        self.update(local_vars)
