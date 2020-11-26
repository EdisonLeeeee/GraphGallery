import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from typing import Union, Optional, List, Tuple, Any

from .hetegraph import HeteGraph


class EdgeGraph(HeteGraph):
    """Attributed labeled heterogeneous graph stored in 
        Numpy array form."""
    multiple = False

    @property
    def A(self):
        raise NotImplementedError

    def to_Graph(self):
        raise NotImplementedError
