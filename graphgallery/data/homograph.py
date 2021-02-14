import numpy as np
import networkx as nx
import scipy.sparse as sp

from typing import Union, Optional, List, Tuple, Any, Callable

from graphgallery import functional as gf

from .base_graph import BaseGraph
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
        return get_num_nodes(self.adj_matrix)

    @property
    def num_edges(self) -> int:
        return get_num_edges(self.adj_matrix, self.is_directed())

    @property
    def density(self) -> float:
        """Get the density of the graph.
        It is defined as M/(N x N)
        where M is number of edges and N is number of nodes
        """
        return self.num_edges / self.num_nodes**2

    @property
    def sparsity(self) -> float:
        """Get the sparsity of the graph.
        It is defined as 1 - M/(N x N)
        where M is number of edges and N is number of nodes
        """
        return 1.0 - self.density

    @property
    def num_graphs(self) -> int:
        return get_num_graphs(self.adj_matrix)

    @property
    def num_node_attrs(self) -> int:
        return get_num_node_attrs(self.node_attr)

    @property
    def num_node_classes(self) -> int:
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
        return gf.degree(self.adj_matrix)

    def eliminate_selfloops(self):
        """Remove self-loops from the adjacency matrix."""
        g = self.copy()
        A = g.adj_matrix
        assert A is not None
        g.adj_matrix = gf.eliminate_selfloops(A)
        return g

    def add_selfloops(self, fill_weight=1.0):
        g = self.copy()
        A = g.adj_matrix
        assert A is not None
        g.adj_matrix = gf.add_selfloops(A, fill_weight=fill_weight)
        return g

    def is_directed(self) -> bool:
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return gf.is_directed(self.adj_matrix)

    def has_singleton(self) -> bool:
        """Check if the graph has singletons."""
        return gf.has_singleton(self.adj_matrix)

    def has_selfloops(self) -> bool:
        """Check if the graph has self loops."""
        return gf.has_selfloops(self.adj_matrix)

    def is_connected(self) -> bool:
        """Check if the graph is a connected graph."""
        return gf.is_connected(self.adj_matrix)

    def is_weighted(self) -> bool:
        """Check if the graph is weighted (edge weights other than 1)."""
        return gf.is_weighted(self.adj_matrix)

    def is_symmetric(self) -> bool:
        """Check if the adjacency matrix is symmetric."""
        return gf.is_symmetric(self.adj_matrix)

    def is_binary(self) -> bool:
        """Check if the node attribute matrix has binary attributes."""
        return gf.is_binary(self.node_attr)

    def extra_repr(self) -> str:
        excluded = {"metadata", "mapping"}
        string = ""
        blank = ' ' * (len(self.__class__.__name__) + 1)
        for k, v in self.items():
            if v is None or k in excluded:
                continue
            string += f"{k}{getattr(v, 'shape')},\n{blank}"
        return string
