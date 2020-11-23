import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from functools import partial
from copy import copy as copy_fn

from typing import Union, Optional, List, Tuple, Any

from .base_graph import BaseGraph
from ..data_type import is_intscalar


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
        collates = locals()
        del collates['self']
        self.update(**collates)

    def extra_repr(self):
        excluded = {"metadata"}
        string = ""
        blank = ' ' * (len(self.__class__.__name__) + 1)
        for k, v in self.items():
            if v is None or k in excluded:
                continue
            string += f"{k}(num={len(v)}),\n{blank}"
        return string

    def __getitem__(self, index):
        if isinstance(index, str):
            return super().__getitem__(index)
        else:
            try:
                G = self.copy()
                collate_fn = partial(index_select, index=index)
                G.update(**G.dicts(collate_fn=collate_fn))
                return G
            except IndexError as e:
                raise IndexError(f"Invalid index {index}.")


def index_select(key, value, index):
    if isinstance(value, np.ndarray):
        value = value[index]
    return key, value
