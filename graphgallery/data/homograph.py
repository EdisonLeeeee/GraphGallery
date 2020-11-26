import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp


from typing import Union, Optional, List, Tuple, Any, Callable

from .base_graph import BaseGraph
from .preprocess import largest_connected_components, create_subgraph

from ..data_type import is_multiobjects


class HomoGraph(BaseGraph):
    """Homogeneous graph stored in sparse matrix form."""
    multiple = False

    def __init__(self, adj_matrix=None,
                 node_attr=None,
                 node_label=None, *,
                 node_graph_label=None,
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
        node_graph_label: np.ndarray, shape [num_nodes], optional
            Array, representing the node belongs to which graph
        graph_attr: sp.csr_matrix or np.ndarray, optional.
            Graph attribute matrix in CSR or Numpy format.
        graph_label: np.ndarray, optional
            Array, graph label matrix
        mapping: dict, optional
            Mapping objects
        metadata : dict, optional
            Additional metadata such as text.
        copy: bool, optional
            whether to use copy for the inputs.
        """
        collects = locals()
        del collects['self']
        super().__init__(**collects)

    @ property
    def num_nodes(self) -> int:
        """Get the number of nodes in the graph."""
        return get_num_nodes(self.adj_matrix)

    @property
    def num_edges(self):
        """Get the number of edges in the graph.
        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        return get_num_edges(self.adj_matrix, self.is_directed())

    @ property
    def num_graphs(self) -> int:
        """Get the number of graphs."""
        return get_num_graphs(self.adj_matrix)

    @ property
    def num_node_attrs(self) -> int:
        """Get the number of attribute dimensions of the nodes."""
        return get_num_node_attrs(self.node_attr)

    @ property
    def num_node_classes(self) -> int:
        """Get the number of classes node_label of the nodes."""
        return get_num_node_classes(self.node_label)

    @property
    def A(self):
        """alias of adj_matrix."""
        return self.adj_matrix

    @property
    def x(self):
        """alias of node_attr."""
        return self.node_attr

    @property
    def y(self):
        """alias of node_label."""
        return self.node_label

    @property
    def d(self):
        """alias of degrees."""
        return self.degrees

    def is_directed(self) -> bool:
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        adj_matrix = self.adj_matrix
        if is_multiobjects(adj_matrix):
            adj_matrix = adj_matrix[0]
        return (adj_matrix != adj_matrix.T).sum() != 0

    @ property
    def degrees(self):
        assert self.adj_matrix is not None

        if not self.is_directed():
            return self.adj_matrix.sum(1).A1
        else:
            # in-degree and out-degree
            return self.adj_matrix.sum(0).A1, self.adj_matrix.sum(1).A1

    def neighbors(self, idx):
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.
        """
        return self.adj_matrix[idx].indices

    def to_undirected(self):
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

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        G = self.copy()
        A = G.adj_matrix
        G.adj_matrix = sp.csr_matrix(
            (np.ones_like(A.data), A.indices, A.indptr), shape=A.shape)
        return G

    def eliminate_selfloops(self):
        """Remove self-loops from the adjacency matrix."""
        G = self.copy()
        A = G.adj_matrix
        A = A - sp.diags(A.diagonal())
        A.eliminate_zeros()
        G.adj_matrix = A
        return G

    def eliminate_classes(self, threshold=0):
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

    def eliminate_singleton(self):
        G = self.graph.eliminate_selfloops()
        A = G.adj_matrix
        mask = np.logical_and(A.sum(0) == 0, A.sum(1) == 1)
        nodes_to_keep = mask.nonzero()[0]
        return G.subgraph(nodes_to_keep=nodes_to_keep)

    def add_selfloops(self, value=1.0):
        """Set the diagonal."""
        G = self.eliminate_selfloops()
        A = G.adj_matrix
        A = A + sp.diags(A.diagonal() + value)
        A.eliminate_zeros()
        G.adj_matrix = A
        return G

    def standardize(self):
        """Select the LCC of the unweighted/undirected/no-self-loop graph.
        All changes are done inplace.
        """
        G = self.to_unweighted().to_undirected().eliminate_selfloops()
        G = largest_connected_components(G, 1)
        return G

    def nxgraph(self, directed: bool = True):
        """Get the network graph from adj_matrix."""
        if directed:
            create_using = nx.DiGraph
        else:
            create_using = nx.Graph
        return nx.from_scipy_sparse_matrix(self.adj_matrix, create_using=create_using)

    def subgraph(self, *, nodes_to_remove=None, nodes_to_keep=None):
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

    def extra_repr(self) -> str:
        excluded = {"metadata"}
        string = ""
        blank = ' ' * (len(self.__class__.__name__) + 1)
        for k, v in self.items():
            if v is None or k in excluded:
                continue
            string += f"{k}{getattr(v, 'shape')},\n{blank}"
        return string


def get_num_nodes(adj_matrices, fn=sum):
    if adj_matrices is None:
        return 0

    if is_multiobjects(adj_matrices):
        return fn(get_num_nodes(adj_matrix) for adj_matrix in adj_matrices)
        # # NOTE: please make sure all the graph are the same!!
        # return max(get_num_nodes(adj_matrix) for adj_matrix in adj_matrices)

    return adj_matrices.shape[0]


def get_num_graphs(adj_matrices, fn=None):
    if adj_matrices is None:
        return 0

    if is_multiobjects(adj_matrices):
        # return sum(get_num_graphs(adj_matrix) for adj_matrix in adj_matrices)
        return len(adj_matrices)

    return 1


def get_num_edges(adj_matrices, is_directed=False, fn=sum):
    if adj_matrices is None:
        return 0

    if is_multiobjects(adj_matrices):
        return fn(get_num_edges(adj_matrix) for adj_matrix in adj_matrices)

    if is_directed:
        return int(adj_matrices.nnz)
    else:
        A = adj_matrices
        num_diag = (A.diagonal() != 0).sum()
        return int((A.nnz - num_diag) / 2) + int(num_diag)


def get_num_node_attrs(node_attrs, fn=max):
    if node_attrs is None:
        return 0

    if is_multiobjects(node_attrs):
        return fn(get_num_node_attrs(node_attr) for node_attr in node_attrs)

    return node_attrs.shape[1]


def get_num_node_classes(node_labels, fn=max):
    if node_labels is None:
        return 0

    if is_multiobjects(node_labels):
        return fn(get_num_node_classes(node_label) for node_label in node_labels)

    if node_labels.ndim == 1:
        return node_labels.max() + 1
    else:
        return node_labels.shape[1]
