import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from functools import partial
from copy import copy as copy_fn

from typing import Union, Optional, List, Tuple, Any

from .homograph import HomoGraph
from .graph import Graph
from .apply import index_select
from ..data_type import is_intscalar


class MultiGraph(HomoGraph):
    """Multiple attributed labeled homogeneous graph stored in a list of 
        sparse matrices form."""
    multiple = True

    def __init__(self, adj_matrix=None,
                 node_attr=None,
                 node_label=None, *,
                 edge_attr=None,
                 edge_label=None,
                 graph_attr=None,
                 graph_label=None,
                 mapping=None,
                 metadata: Any = None,
                 copy: bool = True):
        r"""Create a multiple (un)dirtected (attributed and labeled) graph."""
        collects = locals()
        del collects['self']
        self.update(**collects)

    def extra_repr(self):
        excluded = {"metadata", "mapping"}
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
                apply_fn = partial(index_select, index=index)
                collects = self.dicts(apply_fn=apply_fn)
                if is_intscalar(index):
                    # Single graph
                    return Graph(**collects)
                else:
                    G = self.copy()
                    G.update(**collects)
                    return G
            except IndexError as e:
                raise IndexError(f"Invalid index {index}.")
