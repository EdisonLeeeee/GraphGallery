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

    def to_undirected(self):
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        raise NotImplementedError

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        raise NotImplementedError

    def eliminate_selfloops(self):
        """Remove self-loops from the adjacency matrix."""
        raise NotImplementedError

    def eliminate_classes(self, threshold=0):
        """Remove nodes from graph that correspond to a class of which there are less
        or equal than 'threshold'. Those classes would otherwise break the training procedure.
        """
        raise NotImplementedError

    def eliminate_singleton(self):
        raise NotImplementedError

    def add_selfloops(self, value=1.0):
        """Set the diagonal."""
        raise NotImplementedError

    def standardize(self):
        """Select the largest connected components (LCC) of 
        the unweighted/undirected/no-self-loop graph."""
        raise NotImplementedError

    def to_networkx(self, directed: bool = True):
        """Get the network graph from adj_matrix."""
        raise NotImplementedError

    def subgraph(self, *, nodes_to_remove=None, nodes_to_keep=None):
        raise NotImplementedError

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
