import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from functools import lru_cache
from collections import Counter
from copy import copy as copy_fn

from typing import Union, Optional, List, Tuple, Any
from graphgallery.data.base_graph import BaseGraph
from graphgallery.data.preprocess import largest_connected_components, create_subgraph

NxGraph = Union[nx.Graph, nx.DiGraph]
Array1D = Union[List, np.ndarray]
Matrix2D = Union[List[List], np.ndarray]
LabelMatrix = Union[Array1D, Matrix2D]
AdjMatrix = Union[sp.csr_matrix, sp.csc_matrix]


def _check(adj_matrix: Optional[AdjMatrix] = None,
           node_attr: Optional[Matrix2D] = None,
           node_labels: Optional[LabelMatrix] = None,
           copy: bool = True):
    # Make sure that the dimensions of matrices / arrays all agree
    if adj_matrix is not None:
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr(
                copy=False).astype(np.float32, copy=copy)
        else:
            raise ValueError(f"Adjacency matrix must be in sparse format (got {type(adj_matrix)} instead).")

        assert adj_matrix.ndim == 2

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree!")

    if node_attr is not None:
        if sp.isspmatrix(node_attr):
            node_attr = node_attr.tocsr(
                copy=False).astype(np.float32, copy=copy)
        elif isinstance(node_attr, np.ndarray):
            node_attr = node_attr.astype(np.float32, copy=copy)
        else:
            raise ValueError(
                f"Node attribute matrix must be a sp.spmatrix or a np.ndarray (got {type(node_attr)} instead).")

        assert node_attr.ndim == 2

    if node_labels is not None:
        node_labels = np.array(node_labels, dtype=np.int64, copy=copy).squeeze()

        assert 0 < node_labels.ndim <= 2

    return adj_matrix, node_attr, node_labels


class Graph(BaseGraph):
    """Attributed labeled graph stored in sparse matrix form."""

    def __init__(self, adj_matrix: Optional[AdjMatrix] = None,
                 node_attr: Optional[Union[AdjMatrix, Matrix2D]] = None,
                 node_labels: Optional[LabelMatrix] = None,
                 metadata: Any = None,
                 copy: bool = True):
        r"""Create an (un)dirtected (attributed and labeled) graph.

        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [num_nodes, num_nodes], optional
            Adjacency matrix in CSR format.
        node_attr : sp.csr_matrix or np.ndarray, shape [num_nodes, num_node_attrs], optional
            Node attribute matrix in CSR or Numpy format.
        node_labels : np.ndarray, shape [num_nodes], optional
            Array, where each entry represents respective node's label(s).
        metadata : object, optional
            Additional metadata such as text.
        copy: bool, optional
            whether to use copy for the inputs.
        """
        adj_matrix, node_attr, node_labels = _check(
            adj_matrix, node_attr, node_labels, copy=copy)

        local_vars = locals()
        del local_vars['self'], local_vars['copy']
        self.update(local_vars)

    @ property
    def degrees(self) -> Union[Tuple[Array1D, Array1D], Array1D]:
        if not self.is_directed():
            return self.adj_matrix.sum(1).A1
        else:
            # in-degree and out-degree
            return self.adj_matrix.sum(0).A1, self.adj_matrix.sum(1).A1

    @ property
    def num_nodes(self) -> int:
        """Get the number of nodes in the graph."""
        if self.adj_matrix is not None:
            return self.adj_matrix.shape[0]
        else:
            return None

    @ property
    def num_edges(self) -> int:
        """Get the number of edges in the graph.
        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            A = self.adj_matrix
            diag = A.diagonal()
            A = A - sp.diags(diag)
            A.eliminate_zeros()
            return int(A.nnz / 2) + int((diag != 0).sum())

    @ property
    def num_graphs(self) -> int:
        """Get the number of graphs."""
        return 1

    @ property
    def num_node_classes(self) -> int:
        """Get the number of classes node_labels of the nodes."""
        if self.node_labels is not None:
            return self.node_labels.max() + 1
        else:
            return None

    @ property
    def num_node_attrs(self) -> int:
        """Get the number of attribute dimensions of the nodes."""
        if self.node_attr is not None:
            return self.node_attr.shape[1]
        else:
            return None

    @ property
    def node_labels_onehot(self) -> Matrix2D:
        """Get the one-hot like node_labels of nodes."""
        node_labels = self.node_labels
        if node_labels is not None and node_labels.ndim == 1:
            return np.eye(self.num_node_classes)[node_labels].astype(node_labels.dtype)
        return node_labels

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
        if self.node_labels is None:
            return self
        node_labels = self.node_labels
        counts = np.bincount(node_labels)
        nodes_to_remove = []
        removed = 0
        left = []
        for _class, count in enumerate(counts):
            if count <= threshold:
                nodes_to_remove.extend(np.where(node_labels == _class)[0])
                removed += 1
            else:
                left.append(_class)

        if removed > 0:
            G = self.subgraph(nodes_to_remove=nodes_to_remove)
            mapping = dict(zip(left, range(self.num_node_classes - removed)))
            G.node_labels = np.asarray(list(map(lambda key: mapping[key], G.node_labels)), dtype=np.int32)
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

    def to_npz(self, filepath):
        filepath = save_graph_to_npz(filepath, self)
        print(f"save to {filepath}.")

    @ staticmethod
    def from_npz(filepath) -> "Graph":
        return load_dataset(filepath)

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

    def is_directed(self) -> bool:
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0


def load_dataset(data_path: str) -> "Graph":
    """Load a dataset.

    Parameters
    ----------
    name : str
        Name of the dataset to load.

    Returns
    -------
    graph : Graph
        The requested dataset in sparse format.
    """
    data_path = osp.abspath(osp.expanduser(osp.realpath(data_path)))

    if not data_path.endswith('.npz'):
        data_path = data_path + '.npz'
    if osp.isfile(data_path):
        return load_npz_to_graph(data_path)
    else:
        raise ValueError(f"{data_path} doesn't exist.")


def load_npz_to_graph(filename: str) -> "Graph":
    """Load a Graph from a Numpy binary file.

    Parameters
    ----------
    filename : str
        Name of the file to load.
    graph : graphgalllery.data.Graph
        Graph in sparse matrix format.
    Returns
    -------
    graph : Graph
        Graph in sparse matrix format.
    """

    with np.load(filename, allow_pickle=True) as loader:
        loader = dict(loader)

        for k, v in loader.items():
            if v.dtype.kind == 'O':
                loader[k] = v.tolist()

        return Graph(**loader)


def save_graph_to_npz(filepath: str, graph: "Graph"):
    """Save a Graph to a Numpy binary file.

    Parameters
    ----------
    filepath : str
        Name of the output file.
    graph : graphgalllery.data.Graph
        Graph in sparse matrix format.
    """
    data_dict = {k: v for k, v in graph.items if v is not None}

    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'

    filepath = osp.abspath(osp.expanduser(osp.realpath(filepath)))
    np.savez_compressed(filepath, **data_dict)
    return filepath
