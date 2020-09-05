import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from functools import lru_cache
from copy import copy as copy_fn

from graphgallery.data.preprocess import largest_connected_components
from graphgallery.data.basegraph import BaseGraph


def _check_and_convert(adj_matrix, attr_matrix, labels, copy=True):
    # Make sure that the dimensions of matrices / arrays all agree
    if adj_matrix is not None:
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr(copy=False).astype(np.float32, copy=copy)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)"
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree!")

    if attr_matrix is not None:
        if sp.isspmatrix(attr_matrix):
            attr_matrix = attr_matrix.tocsr(copy=False).astype(np.float32, copy=copy)
        elif isinstance(attr_matrix, np.ndarray):
            attr_matrix = attr_matrix.astype(np.float32, copy=copy)
        else:
            raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)".format(type(attr_matrix)))

        if attr_matrix.shape[0] != adj_matrix.shape[0]:
            raise ValueError("Dimensions of the adjacency and attribute matrices don't agree!")

    if labels is not None:
        labels = np.array(labels, dtype=np.int32, copy=copy)
        if labels.ndim != 1:
            labels = labels.argmax(1)
        # if labels.shape[0] != adj_matrix.shape[0]:
            # raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree!")

    return adj_matrix, attr_matrix, labels


class Graph(BaseGraph):
    """Attributed labeled graph stored in sparse matrix form."""

    # By default, the attr_matrix is dense format, i.e., Numpy array
    _sparse_attr = False

    def __init__(self, adj_matrix=None, attr_matrix=None, labels=None,
                 node_names=None, attr_names=None, class_names=None, metadata=None, copy=True):
        """Create an (un)dirtected (attributed and labeled) graph.

        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [n_nodes, n_nodes], optional
            Adjacency matrix in CSR format.
        attr_matrix : sp.csr_matrix or np.ndarray, shape [n_nodes, n_attrs], optional
            Attribute matrix in CSR or Numpy format.
        labels : np.ndarray, shape [n_nodes], optional
            Array, where each entry represents respective node's label(s).
        node_names : np.ndarray, shape [n_nodes], optional
            Names of nodes (as strings).
        attr_names : np.ndarray, shape [n_attrs]
            Names of the attributes (as strings).
        class_names : np.ndarray, shape [n_classes], optional
            Names of the class labels (as strings).
        metadata : object, optional
            Additional metadata such as text.
        copy: bool, optional
            whether to use copy for the inputs.
        """
        adj_matrix, attr_matrix, labels = _check_and_convert(adj_matrix, attr_matrix, labels, copy=copy)

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree!")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree!")

        self._adj_matrix = adj_matrix
        self._attr_matrix = attr_matrix
        self._labels = labels

        if copy:
            self.node_names = copy_fn(node_names)
            self.attr_names = copy_fn(attr_names)
            self.class_names = copy_fn(class_names)
            self.metadata = copy_fn(metadata)
        else:
            self.node_names = node_names
            self.attr_names = attr_names
            self.class_names = class_names
            self.metadata = metadata

    def set_inputs(self, adj_matrix, attr_matrix=None, labels=None, copy=True):
        adj_matrix, attr_matrix, labels = _check_and_convert(adj_matrix, attr_matrix,
                                                             labels, copy=copy)

        if adj_matrix is not None:
            self._adj_matrix = adj_matrix

        if attr_matrix is not None:
            self._attr_matrix = attr_matrix
            # clear LRU cache
            self.get_attr_matrix.cache_clear()

        if labels is not None:
            self._labels = labels

    def from_inputs(self, adj_matrix=None, attr_matrix=None, labels=None, copy=True):
        if adj_matrix is None:
            adj_matrix = self._adj_matrix
        if attr_matrix is None:
            attr_matrix = self._attr_matrix
        if labels is None:
            labels = self._labels

        return Graph(adj_matrix=adj_matrix,
                     attr_matrix=attr_matrix,
                     labels=labels, copy=copy)

    @property
    def sparse_attr(self):
        return self._sparse_attr

    @sparse_attr.setter
    def sparse_attr(self, value):
        self._sparse_attr = value
        self.get_attr_matrix.cache_clear()

    @lru_cache(maxsize=1)
    def get_attr_matrix(self):
        is_sparse = sp.isspmatrix(self._attr_matrix)
        if not self.sparse_attr and is_sparse:
            return self._attr_matrix.A
        elif self.sparse_attr and not is_sparse:
            return sp.csr_matrix(self._attr_matrix)
        return self._attr_matrix

    @property
    def adj_matrix(self):
        return self._adj_matrix

    @property
    def attr_matrix(self):
        return self.get_attr_matrix()

    @property
    def labels(self):
        return self._labels

    @property
    def adj(self):
        """alias of adj_matrix"""
        return self.adj_matrix

    @property
    def x(self):
        """alias of attr_matrix"""
        return self.attr_matrix

    @property
    def y(self):
        """alias of labels"""
        return self.labels

    @property
    def n_nodes(self):
        """Get the number of nodes in the graph."""
        return self.adj_matrix.shape[0]

    @property
    def n_edges(self):
        """Get the number of edges in the graph.
        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    @property
    def n_graphs(self):
        """Get the number of graphs."""
        return 1

    @property
    def n_classes(self):
        """Get the number of classes labels of the nodes."""
        if self.labels is not None:
            return self.labels.max()+1
        else:
            return None

    @property
    def n_attrs(self):
        """Get the number of attribute dimensions of the nodes."""
        if self.attr_matrix is not None:
            return self.attr_matrix.shape[1]
        else:
            return None

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
            raise ValueError("Convert to unweighted graph first.")
        else:
            A = self._adj_matrix
            A = A.maximum(A.T)
            self._adj_matrix = A
        return self

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        self._adj_matrix.data = np.ones_like(self._adj_matrix.data)
        return self

    def eliminate_self_loops(self):
        """Remove self-loops from the adjacency matrix."""
        A = self._adj_matrix
        A -= sp.diags(A.diagonal())
        A.eliminate_zeros()
        self._adj_matrix = A
        return self

    def add_self_loops(self, value=1.0):
        """Set the diagonal."""
        self.eliminate_self_loops()
        A = self._adj_matrix
        A += sp.diags(A.diagonal()+value)
        if value == 0:
            A.eliminate_zeros()
        self._adj_matrix = A
        return self

    def standardize(self):
        """Select the LCC of the unweighted/undirected/no-self-loop graph.
        All changes are done inplace.
        """
        G = self.to_unweighted().to_undirected().eliminate_self_loops()
        G = largest_connected_components(G, 1)
        return G

    def unpack(self):
        """Return the (A, X, Y) triplet."""
        return self.adj_matrix, self.attr_matrix, self.labels

    def nxgraph(self, directed=True):
        """Get the network graph from adj_matrix."""
        if directed:
            create_using = nx.DiGraph
        else:
            create_using = nx.Graph
        return nx.from_scipy_sparse_matrix(self.adj_matrix, create_using=create_using)

    def to_npz(self, filepath):
        filepath = save_sparse_graph_to_npz(filepath, self)
        print(f"save to {filepath}.")

    @staticmethod
    def from_npz(filepath):
        return load_dataset(filepath)

    def copy(self, deepcopy=False):
        new_graph = Graph(*self.unpack(), node_names=self.node_names,
                          attr_names=self.attr_names, class_names=self.class_names,
                          metadata=self.metadata, copy=deepcopy)
#         new_graph.__dict__ = self.__dict__
        return new_graph

    def is_singleton(self):
        """Check if the input adjacency matrix has singletons."""
        A = self.adj_matrix
        out_deg = A.sum(1).A1
        in_deg = A.sum(0).A1
        return np.where(np.logical_and(in_deg == 0, out_deg == 0))[0].size != 0

    def is_self_loops(self):
        '''Check if the input Scipy sparse adjacency matrix has self loops.'''
        return self.adj_matrix.diagonal().sum() != 0

    def is_binary(self):
        '''Check if the attribute matrix has binary attributes.'''
        return np.any(np.unique(self.attr_matrix[self.attr_matrix != 0].A1) != 1)

    def is_weighted(self):
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def is_directed(self):
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def is_labeled(self):
        return self.labels is not None

    def is_attributed(self):
        return self.attr_matrix is not None

    def __repr__(self):
        A_shape = self.adj_matrix.shape if self.adj_matrix is not None else (None,)
        X_shape = self.adj_matrix.shape if self.adj_matrix is not None else (None,)
        Y_shape = self.labels.shape if self.labels is not None else (None,)
        return f"{self.__class__.__name__}(adj_matrix{A_shape}, attr_matrix{X_shape}, labels{Y_shape})"


def load_dataset(data_path):
    """Load a dataset.

    Parameters
    ----------
    name : str
        Name of the dataset to load.

    Returns
    -------
    sparse_graph : Graph
        The requested dataset in sparse format.
    """
    data_path = osp.abspath(osp.expanduser(osp.normpath(data_path)))

    if not data_path.endswith('.npz'):
        data_path = data_path + '.npz'
    if osp.isfile(data_path):
        return load_npz_to_graph(data_path)
    else:
        raise ValueError(f"{data_path} doesn't exist.")


def load_npz_to_graph(file_name):
    """Load a Graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : Graph
        Graph in sparse matrix format.
    """

    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a Numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return Graph(adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata)


def save_sparse_graph_to_npz(filepath, sparse_graph):
    """Save a Graph to a Numpy binary file.

    Parameters
    ----------
    filepath : str
        Name of the output file.
    sparse_graph : graphgalllery.data.Graph
        Graph in sparse matrix format.
    """
    data_dict = {
        'adj_data': sparse_graph.adj_matrix.data,
        'adj_indices': sparse_graph.adj_matrix.indices,
        'adj_indptr': sparse_graph.adj_matrix.indptr,
        'adj_shape': sparse_graph.adj_matrix.shape
    }
    if sp.isspmatrix(sparse_graph.attr_matrix):
        data_dict['attr_data'] = sparse_graph.attr_matrix.data
        data_dict['attr_indices'] = sparse_graph.attr_matrix.indices
        data_dict['attr_indptr'] = sparse_graph.attr_matrix.indptr
        data_dict['attr_shape'] = sparse_graph.attr_matrix.shape
    elif isinstance(sparse_graph.attr_matrix, np.ndarray):
        data_dict['attr_matrix'] = sparse_graph.attr_matrix

    if sp.isspmatrix(sparse_graph.labels):
        data_dict['labels_data'] = sparse_graph.labels.data
        data_dict['labels_indices'] = sparse_graph.labels.indices
        data_dict['labels_indptr'] = sparse_graph.labels.indptr
        data_dict['labels_shape'] = sparse_graph.labels.shape
    elif isinstance(sparse_graph.labels, np.ndarray):
        data_dict['labels'] = sparse_graph.labels

    if sparse_graph.node_names is not None:
        data_dict['node_names'] = sparse_graph.node_names

    if sparse_graph.attr_names is not None:
        data_dict['attr_names'] = sparse_graph.attr_names

    if sparse_graph.class_names is not None:
        data_dict['class_names'] = sparse_graph.class_names

    if sparse_graph.metadata is not None:
        data_dict['metadata'] = sparse_graph.metadata

    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'

    filepath = osp.abspath(osp.expanduser(osp.normpath(filepath)))
    np.savez(filepath, **data_dict)
    return filepath
