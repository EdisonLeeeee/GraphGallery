import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from functools import lru_cache
from collections import Counter
from copy import copy as copy_fn

from typing import Union, Optional, List, Tuple
from graphgallery.data.base_graph import BaseGraph
from graphgallery.data.preprocess import largest_connected_components, create_subgraph

NxGraph = Union[nx.Graph, nx.DiGraph]
Array1D = Union[List, np.ndarray]
Matrix2D = Union[List[List], np.ndarray]
LabelMatrix = Union[Array1D, Matrix2D]
AdjMatrix = Union[sp.csr_matrix, sp.csc_matrix]


def _check(adj_matrix: Optional[AdjMatrix] = None,
           attr_matrix: Optional[Matrix2D] = None,
           labels: Optional[LabelMatrix] = None,
           copy: bool = True):
    # Make sure that the dimensions of matrices / arrays all agree
    if adj_matrix is not None:
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr(
                copy=False).astype(np.float32, copy=copy)
        else:
            raise ValueError(f"Adjacency matrix must be in sparse format (got {type(adj_matrix)} instead).")

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree!")

    if attr_matrix is not None:
        if sp.isspmatrix(attr_matrix):
            attr_matrix = attr_matrix.tocsr(
                copy=False).astype(np.float32, copy=copy)
        elif isinstance(attr_matrix, np.ndarray):
            attr_matrix = attr_matrix.astype(np.float32, copy=copy)
        else:
            raise ValueError(
                f"Attribute matrix must be a sp.spmatrix or a np.ndarray (got {type(attr_matrix)} instead).")

        if adj_matrix is not None and attr_matrix.shape[0] != adj_matrix.shape[0]:
            raise ValueError(
                "Dimensions of the adjacency and attribute matrices don't agree!")

    if labels is not None:
        labels = np.array(labels, dtype=np.int64, copy=copy).squeeze()
        if not 0 < labels.ndim <= 2:
            raise ValueError("Label matrix must be a 1D or 2D array!")

    return adj_matrix, attr_matrix, labels


class Graph(BaseGraph):
    """Attributed labeled graph stored in sparse matrix form."""

    def __init__(self, adj_matrix: Optional[AdjMatrix] = None,
                 attr_matrix: Optional[Union[AdjMatrix, Matrix2D]] = None,
                 labels: Optional[LabelMatrix] = None,
                 metadata: str = None,
                 copy: bool = True):
        """Create an (un)dirtected (attributed and labeled) graph.

        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [n_nodes, n_nodes], optional
            Adjacency matrix in CSR format.
        attr_matrix : sp.csr_matrix or np.ndarray, shape [n_nodes, n_attrs], optional
            Attribute matrix in CSR or Numpy format.
        labels : np.ndarray, shape [n_nodes], optional
            Array, where each entry represents respective node's label(s).
        metadata : object, optional
            Additional metadata such as text.
        copy: bool, optional
            whether to use copy for the inputs.
        """
        adj_matrix, attr_matrix, labels = _check(
            adj_matrix, attr_matrix, labels, copy=copy)

        self._adj_matrix = adj_matrix
        self._attr_matrix = attr_matrix
        self._labels = labels

        if copy:
            self.metadata = copy_fn(metadata)
        else:
            self.metadata = metadata

    def set_inputs(self, adj_matrix: Optional[AdjMatrix] = None,
                   attr_matrix: Optional[Union[AdjMatrix, Matrix2D]] = None,
                   labels: Optional[LabelMatrix] = None,
                   metadata: str = None,
                   copy: bool = False):

        adj_matrix, attr_matrix, labels = _check(adj_matrix, attr_matrix,
                                                 labels, copy=copy)

        if adj_matrix is not None:
            self.adj_matrix = adj_matrix

        if attr_matrix is not None:
            self.attr_matrix = attr_matrix

        if labels is not None:
            self.labels = labels

        if metadata is not None:
            if copy:
                self.metadata = copy_fn(metadata)
            else:
                self.metadata = metadata

    @lru_cache(maxsize=1)
    def get_attr_matrix(self) -> Union[AdjMatrix, Matrix2D]:
        if self._attr_matrix is None:
            n_nodes = self.n_nodes
            if n_nodes:
                return np.eye(n_nodes, dtype=np.float32)
            else:
                return None

        is_sparse = sp.isspmatrix(self._attr_matrix)
        if is_sparse:
            return self._attr_matrix.toarray()
        return self._attr_matrix

    @property
    def adj_matrix(self) -> AdjMatrix:
        return self._adj_matrix

    @adj_matrix.setter
    def adj_matrix(self, x):
        self._adj_matrix = x

    @property
    def attr_matrix(self) -> Union[AdjMatrix, Matrix2D]:
        return self.get_attr_matrix()

    @attr_matrix.setter
    def attr_matrix(self, x):
        # clear LRU cache
        self.get_attr_matrix.cache_clear()
        self._attr_matrix = x

    @property
    def labels(self) -> LabelMatrix:
        _labels = self._labels
        if _labels.ndim == 2 and (_labels.sum(1) == 1).all():
            _labels = _labels.argmax(1)
        return _labels

    @labels.setter
    def labels(self, x):
        self._labels = x

    @property
    def degrees(self) -> Union[Tuple[Array1D, Array1D], Array1D]:
        if not self.is_directed():
            return self.adj_matrix.sum(1).A1
        else:
            # in-degree and out-degree
            return self.adj_matrix.sum(0).A1, self.adj_matrix.sum(1).A1

    @property
    def n_nodes(self) -> int:
        """Get the number of nodes in the graph."""
        if self.adj_matrix is not None:
            return self.adj_matrix.shape[0]
        else:
            return None

    @property
    def n_edges(self) -> int:
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

    @property
    def n_graphs(self) -> int:
        """Get the number of graphs."""
        return 1

    @property
    def n_classes(self) -> int:
        """Get the number of classes labels of the nodes."""
        if self.labels is not None:
            return self.labels.max() + 1
        else:
            return None

    @property
    def n_attrs(self) -> int:
        """Get the number of attribute dimensions of the nodes."""
        if self.attr_matrix is not None:
            return self.attr_matrix.shape[1]
        else:
            return None

    @property
    def labels_onehot(self) -> Matrix2D:
        """Get the one-hot like labels of nodes."""
        labels = self.labels
        if labels is not None:
            return np.eye(self.n_classes)[labels].astype(labels.dtype)

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
        if self.labels is None:
            return self
        labels = self.labels
        counts = np.bincount(labels)
        nodes_to_remove = []
        removed = 0
        left = []
        for _class, count in enumerate(counts):
            if count <= threshold:
                nodes_to_remove.extend(np.where(labels == _class)[0])
                removed += 1
            else:
                left.append(_class)

        if removed > 0:
            G = self.subgraph(nodes_to_remove=nodes_to_remove)
            mapping = dict(zip(left, range(self.n_classes - removed)))
            G.labels = np.asarray(list(map(lambda key: mapping[key], G.labels)), dtype=np.int32)
            return G
        else:
            return self

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

    @staticmethod
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
        '''Check if the attribute matrix has binary attributes.'''
        return np.all(np.unique(self.attr_matrix) == (0, 1))

    def is_weighted(self) -> bool:
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(self.adj_matrix.data != 1)

    def is_directed(self) -> bool:
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def __repr__(self):
        A_shape = self.adj_matrix.shape if self.adj_matrix is not None else (
            None, None)
        X_shape = self.attr_matrix.shape if self.attr_matrix is not None else (
            None, None)
        Y_shape = self.labels.shape if self.labels is not None else (None,)
        return f"{self.__class__.__name__}(adj_matrix{A_shape}, attr_matrix{X_shape}, labels{Y_shape})"


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
    data_path = osp.abspath(osp.expanduser(osp.normpath(data_path)))

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

    Returns
    -------
    graph : Graph
        Graph in sparse matrix format.
    """

    with np.load(filename, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'],
                                    loader['adj_indices'],
                                    loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'],
                                         loader['attr_indices'],
                                         loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        else:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader.get('attr_matrix', None)

        # Labels are stored as a Numpy array
        labels = loader.get('labels', None)
        metadata = loader.get('metadata', None)

    return Graph(adj_matrix, attr_matrix, labels, metadata)


def save_graph_to_npz(filepath: str, graph: "Graph"):
    """Save a Graph to a Numpy binary file.

    Parameters
    ----------
    filepath : str
        Name of the output file.
    graph : graphgalllery.data.Graph
        Graph in sparse matrix format.
    """
    adj_matrix, attr_matrix, labels = graph.raw()

    data_dict = {
        'adj_data': adj_matrix.data,
        'adj_indices': adj_matrix.indices,
        'adj_indptr': adj_matrix.indptr,
        'adj_shape': adj_matrix.shape
    }

    if sp.isspmatrix(attr_matrix):
        data_dict['attr_data'] = attr_matrix.data
        data_dict['attr_indices'] = attr_matrix.indices
        data_dict['attr_indptr'] = attr_matrix.indptr
        data_dict['attr_shape'] = attr_matrix.shape
    elif attr_matrix is not None:
        data_dict['attr_matrix'] = attr_matrix

    if labels is not None:
        data_dict['labels'] = labels

    if graph.metadata is not None:
        data_dict['metadata'] = graph.metadata

    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'

    filepath = osp.abspath(osp.expanduser(osp.realpath(filepath)))
    np.savez_compressed(filepath, **data_dict)
    return filepath
