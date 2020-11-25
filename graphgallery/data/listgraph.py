import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from .base_graph import BaseGraph
from typing import Union, Optional, List, Tuple, Any, Callable


class ListGraph(BaseGraph):
    """A list of Homogeneous or Heterogeneous graph."""
    multiple = False

    def __init__(self, *list_graphs,
                 metadata: Any = None,
                 copy: bool = True):
        collects = locals()
        del collects['self']
        self.update(**collects)

    @property
    def graphs(self):
        return self.list_graphs

    @property
    def g(self):
        return self.graphs

    @property
    def graph(self):
        return self.graphs[0] if self.graphs else None

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        else:
            # it should be 'int' or 'numpy int'
            return self.graphs[key]

    def __len__(self):
        return len(self.graphs)

    def __iter__(self):
        yield from self.graphs

    def extra_repr(self):
        return f"graphs=({len(self)}), "
