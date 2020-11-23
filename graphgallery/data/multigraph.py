import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from functools import lru_cache
from copy import copy as copy_fn

from typing import Union, Optional, List, Tuple, Any

from .base_graph import BaseGraph


class MultiGraph(BaseGraph):
    """Attributed labeled multigraph stored in a list of sparse matrix form."""
    multiple = True

    def __init__(self, adj_matrix=None,
                 node_attr=None,
                 node_labels=None,
                 edge_attr=None,
                 edge_labels=None,
                 graph_labels=None,
                 graph_attr=None,
                 mapping=None,
                 metadata: Any = None,
                 copy: bool = True):
        r"""Create a multiple (un)dirtected (attributed and labeled) graph.

        """
        collects = locals()
        del collects['self']
        self.update(**collects)

    def extra_repr(self):
        excluded = {"metadata"}
        string = ""
        blank = ' ' * (len(self.__class__.__name__) + 1)
        for k, v in self.items():
            if v is None or k in excluded:
                continue
            string += f"{k}(num={len(v)}),\n{blank}"
        return string
