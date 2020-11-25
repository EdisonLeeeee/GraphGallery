import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from functools import lru_cache
from collections import Counter
from copy import copy as copy_fn

from typing import Union, Optional, List, Tuple, Any

from .base_graph import BaseGraph
from .preprocess import largest_connected_components, create_subgraph

NxGraph = Union[nx.Graph, nx.DiGraph]
Array1D = Union[List, np.ndarray]
Matrix2D = Union[List[List], np.ndarray]
ArrOrMatrix = Union[Array1D, Matrix2D]
AdjMatrix = Union[sp.csr_matrix, sp.csc_matrix]


class Graph(BaseGraph):
    """Attributed labeled graph stored in sparse matrix form."""
    multiple = False

    def __init__(self, adj_matrix: Optional[AdjMatrix] = None,
                 node_attr: Optional[Union[AdjMatrix, Matrix2D]] = None,
                 node_label: Optional[ArrOrMatrix] = None, *,
                 edge_attr=None,
                 edge_label=None,
                 graph_attr=None,
                 graph_label=None,
                 mapping=None,
                 metadata: Any = None,
                 copy: bool = False):
        r"""Create an (un)dirtected (attributed and labeled) graph.

        Parameters
        ----------
        adj_matrix : sp.csr_matrix, 
            shape [num_nodes, num_nodes], optional
            Adjacency matrix in CSR format.
        node_attr : sp.csr_matrix or np.ndarray, 
            shape [num_nodes, num_node_attrs], optional
            Node attribute matrix in CSR or Numpy format.
        node_label : np.ndarray, shape [num_nodes], optional
            Array, where each entry represents respective node's label(s).
        edge_attr:
        edge_label:
        graph_attr:
        graph_label:
        mapping:
        metadata : object, optional
            Additional metadata such as text.
        copy: bool, optional
            whether to use copy for the inputs.
        """
        collects = locals()
        del collects['self']
        self.update(**collects)

    @ property
    def degrees(self) -> Union[Tuple[Array1D], Array1D]:
        assert self.adj_matrix is not None

        if not self.is_directed():
            return self.adj_matrix.sum(1).A1
        else:
            # in-degree and out-degree
            return self.adj_matrix.sum(0).A1, self.adj_matrix.sum(1).A1

    @ property
    def node_label_onehot(self) -> Matrix2D:
        """Get the one-hot like node_label of nodes."""
        node_label = self.node_label
        if node_label is not None and node_label.ndim == 1:
            return np.eye(self.num_node_classes, dtype=node_label.dtype)[node_label]
        return node_label

    def neighbors(self, idx) -> Array1D:
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.
        """
        return self.adj_matrix[idx].indices

    def to_undirected(self) -> "Graph":
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        if self.is_weighted():
            raise ValueError(
                "Convert to unweighted graph first. Using 'graph.to_unweighted()'.")
        else:
            G = self.copy()
            A = G.adj_matrix
            A = A.maximum(A.T)
            G.adj_matrix = A
        return G

    def to_unweighted(self) -> "Graph":
        """Convert to an unweighted graph (set all edge weights to 1)."""
        G = self.copy()
        A = G.adj_matrix
        G.adj_matrix = sp.csr_matrix(
            (np.ones_like(A.data), A.indices, A.indptr), shape=A.shape)
        return G

    def eliminate_selfloops(self) -> "Graph":
        """Remove self-loops from the adjacency matrix."""
        G = self.copy()
        A = G.adj_matrix
        A = A - sp.diags(A.diagonal())
        A.eliminate_zeros()
        G.adj_matrix = A
        return G

    def eliminate_classes(self, threshold=0) -> "Graph":
        """Remove nodes from graph that correspond to a class of which there are less
        or equal than 'threshold'. Those classes would otherwise break the training procedure.
        """
        if self.node_label is None:
            return self
        node_label = self.node_label
        counts = np.bincount(node_label)
        nodes_to_remove = []
        removed = 0
        left = []
        for _class, count in enumerate(counts):
            if count <= threshold:
                nodes_to_remove.extend(np.where(node_label == _class)[0])
                removed += 1
            else:
                left.append(_class)

        if removed > 0:
            G = self.subgraph(nodes_to_remove=nodes_to_remove)
            mapping = dict(zip(left, range(self.num_node_classes - removed)))
            G.node_label = np.asarray(list(map(lambda key: mapping[key], G.node_label)), dtype=np.int32)
            return G
        else:
            return self

    def eliminate_singleton(self) -> "Graph":
        G = self.graph.eliminate_selfloops()
        A = G.adj_matrix
        mask = np.logical_and(A.sum(0) == 0, A.sum(1) == 1)
        nodes_to_keep = mask.nonzero()[0]
        return G.subgraph(nodes_to_keep=nodes_to_keep)

    def add_selfloops(self, value=1.0) -> "Graph":
        """Set the diagonal."""
        G = self.eliminate_selfloops()
        A = G.adj_matrix
        A = A + sp.diags(A.diagonal() + value)
        A.eliminate_zeros()
        G.adj_matrix = A
        return G

    def standardize(self) -> "Graph":
        """Select the LCC of the unweighted/undirected/no-self-loop graph.
        All changes are done inplace.
        """
        G = self.to_unweighted().to_undirected().eliminate_selfloops()
        G = largest_connected_components(G, 1)
        return G

    def nxgraph(self, directed: bool = True) -> NxGraph:
        """Get the network graph from adj_matrix."""
        if directed:
            create_using = nx.DiGraph
        else:
            create_using = nx.Graph
        return nx.from_scipy_sparse_matrix(self.adj_matrix, create_using=create_using)

    def subgraph(self, *, nodes_to_remove=None, nodes_to_keep=None) -> "Graph":
        return create_subgraph(self, nodes_to_remove=nodes_to_remove, nodes_to_keep=nodes_to_keep)

    def is_singleton(self) -> bool:
        """Check if the input adjacency matrix has singletons."""
        A = self.adj_matrix
        out_deg = A.sum(1).A1
        in_deg = A.sum(0).A1
        return np.where(np.logical_and(in_deg == 0, out_deg == 0))[0].size != 0

    def is_selfloops(self) -> bool:
        '''Check if the input Scipy sparse adjacency matrix has self loops.'''
        return self.adj_matrix.diagonal().sum() != 0

    def is_binary(self) -> bool:
        '''Check if the node attribute matrix has binary attributes.'''
        return np.all(np.unique(self.node_attr) == (0, 1))

    def is_weighted(self) -> bool:
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(self.adj_matrix.data != 1)

    def extra_repr(self):
        excluded = {"metadata"}
        string = ""
        blank = ' ' * (len(self.__class__.__name__) + 1)
        for k, v in self.items():
            if v is None or k in excluded:
                continue
            string += f"{k}{getattr(v, 'shape')},\n{blank}"
        return string
