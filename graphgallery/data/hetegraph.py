import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from typing import Union, Optional, List, Tuple, Any

from .base_graph import BaseGraph
from .preprocess import largest_connected_components, create_subgraph


class HeteGraph(BaseGraph):
    """Heterogeneous graph stored in numpy array form."""
    multiple = False

    def __init__(self, edge_index, edge_weight,
                 edge_attr, edge_label, *,
                 node_attr, node_label,
                 node_graph_label=None,
                 graph_attr=None,
                 graph_label=None,
                 mapping=None,
                 metadata: Any = None,
                 copy: bool = False):
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
            string += f"{k}{getattr(v, 'shape')},\n{blank}"
        return string
