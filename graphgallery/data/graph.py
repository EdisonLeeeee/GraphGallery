import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from typing import Union, Optional, List, Tuple, Any

from .homograph import HomoGraph
from .preprocess import largest_connected_components, create_subgraph


class Graph(HomoGraph):
    """Attributed labeled homogeneous graph stored in sparse matrix form."""
    multiple = False

    def to_EdgeGraph(self):
        raise NotImplementedError
