import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from typing import Union, Optional, List, Tuple, Any, Callable
from .base_graph import BaseGraph
from graphgallery import functional as gf


class DictGraph(BaseGraph):
    """A dict of Homogeneous or Heterogeneous graph."""
    multiple = False

    def __init__(self, metadata: Any = None,
                 copy: bool = True,
                 **dict_graphs):
        dict_graphs = gf.Bunch(**dict_graphs)
        collects = locals()
        del collects['self']
        super().__init__(**collects)

    @property
    def graphs(self):
        return self.dict_graphs

    @property
    def g(self):
        return self.graphs

    def __len__(self):
        return len(self.graphs.keys()) if self.graphs else 0

    def __getitem__(self, key):
        # to avoid value is 'None'
        v = self.graphs.get(key, "NAN")
        if v != "NAN":
            return v
        else:
            return super().__getitem__(key)

    def __iter__(self):
        yield from self.graphs.items()

    def extra_repr(self):
        return f"graphs={tuple(self.graphs.keys())}, "
