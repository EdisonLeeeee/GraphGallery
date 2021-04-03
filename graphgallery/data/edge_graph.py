import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from .hetegraph import HeteGraph


class EdgeGraph(HeteGraph):
    """Attributed labeled heterogeneous graph stored in 
        Numpy array form."""
    multiple = False

    def to_Graph(self):
        # TODO
        raise NotImplementedError
