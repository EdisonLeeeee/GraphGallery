import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from collections import Counter
from copy import copy as copy_fn

from typing import Union, Optional, List, Tuple, Any
from .hetegraph import HeteGraph


class MultiEdgeGraph(HeteGraph):
    """Multiple attributed labeled heterogeneous graph stored in 
        Numpy array form."""
    multiple = True
