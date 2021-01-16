import numpy as np
import networkx as nx
import scipy.sparse as sp

from typing import Union, Optional, List, Tuple, Any, Callable

import graphgallery.functional as gf

from .base_graph import BaseGraph
from .preprocess import create_subgraph
from .utils import *


class HomoGraph(BaseGraph):
    """Homogeneous graph stored in sparse matrix form."""
    multiple = False

    def __init__(self,
                 adj_matrix=None,
                 node_attr=None,
                 node_label=None,
                 *,
                 node_graph_label=None,
                 graph_attr=None,
                 graph_label=None,
                 mapping=None,
                 metadata: Any = None,
                 copy: bool = False):
        r"""Create an (un)dirtected (attributed and labeled) graph.

        Parameters
        ----------
        adj_matrix: single (Graph) or a list of (MultiGraph) 
            sp.csr_matrix, optional.
            shape [num_nodes, num_nodes] or 
            shape [num_graphs, num_nodes, num_nodes].            
            adjacency matrix in CSR format.
        node_attr: single (Graph) or a list of (MultiGraph) 
            sp.csr_matrix or np.ndarray, optional
            shape [num_nodes, num_node_attrs] or
            shape [num_graphs, num_nodes, num_node_attrs].            
            Node attribute matrix in CSR or Numpy format.
        node_label: single (Graph) or a list of (MultiGraph) 
            np.ndarray, optional.
            shape [num_nodes] or shape [num_graphs, num_nodes].            
            where each entry represents respective node's label(s).
        node_graph_label: single (Graph) or a list of (MultiGraph) 
            np.ndarray, optional
            shape [num_nodes] or shape [num_graphs, num_nodes].
            representing the node belongs to which graph
        graph_attr: single (Graph) or a list of (MultiGraph) np.ndarray, optional.
            graph attribute matrix in CSR or Numpy format.
        graph_label: np.ndarray, optional
            graph label matrix
        mapping: dict, optional
            mapping objects
        metadata : dict, optional
            additional metadata such as text.
        copy: bool, optional
            whether to use copy for the inputs.
        """
        collects = locals()
        del collects['self']
        super().__init__(**collects)

    @property
    def num_nodes(self) -> int:
        """Get the number of nodes in the graph."""
        return get_num_nodes(self.adj_matrix)

    @property
    def num_edges(self) -> int:
        """Get the number of edges in the graph.
        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        return get_num_edges(self.adj_matrix, self.is_directed())

    @property
    def num_graphs(self) -> int:
        """Get the number of graphs."""
        return get_num_graphs(self.adj_matrix)

    @property
    def num_node_attrs(self) -> int:
        """Get the number of attribute dimensions of the nodes."""
        return get_num_node_attrs(self.node_attr)

    @property
    def num_node_classes(self) -> int:
        """Get the number of classes node_label of the nodes."""
        return get_num_node_classes(self.node_label)

    @property
    def num_attrs(self) -> int:
        """Alias of num_node_attrs."""
        return self.num_node_attrs

    @property
    def num_classes(self) -> int:
        """Alias of num_node_classes."""
        return self.num_node_classes

    @property
    def A(self):
        """Alias of adj_matrix."""
        return self.adj_matrix

    @property
    def x(self):
        """Alias of node_attr."""
        return self.nx

    @property
    def nx(self):
        """Alias of node_attr."""
        return self.node_attr

    @property
    def gx(self):
        """Alias of graph_attr."""
        return self.graph_attr

    @property
    def y(self):
        """Alias of node_label."""
        return self.ny

    @property
    def ny(self):
        """Alias of node_label."""
        return self.node_label

    @property
    def gy(self):
        """Alias of graph_label."""
        return self.graph_label

    @property
    def d(self):
        """Alias of degree."""
        return self.degree

    @property
    def degree(self):
        return get_degree(self.adj_matrix)
    
    def add_selfloops(self, fill_weight=1.0):
        g = self.copy()
        A = g.adj_matrix
        assert A is not None
        self.adj_matrix = gf.add_selfloops(A, fill_weight=fill_weight)
        return self

    def is_directed(self) -> bool:
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return is_directed(self.adj_matrix)

    def is_singleton(self) -> bool:
        """Check if the graph has singletons."""
        return is_singleton(self.adj_matrix)

    def is_selfloops(self) -> bool:
        """Check if the graph has self loops."""
        return is_selfloops(self.adj_matrix)

    def is_connected(self) -> bool:
        """Check if the graph is a connected graph."""
        return is_connected(self.adj_matrix)

    def is_weighted(self) -> bool:
        """Check if the graph is weighted (edge weights other than 1)."""
        return is_weighted(self.adj_matrix)

    def is_symmetric(self) -> bool:
        """Check if the adjacency matrix is symmetric."""
        return is_symmetric(self.adj_matrix)

    def is_binary(self) -> bool:
        """Check if the node attribute matrix has binary attributes."""
        return is_binary(self.node_attr)

    def extra_repr(self) -> str:
        excluded = {"metadata", "mapping"}
        string = ""
        blank = ' ' * (len(self.__class__.__name__) + 1)
        for k, v in self.items():
            if v is None or k in excluded:
                continue
            string += f"{k}{getattr(v, 'shape')},\n{blank}"
        return string
