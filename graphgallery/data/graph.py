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
from graphgallery.typing import SparseMatrix, ArrayLike1D, ArrayLike2D, NxGraph, GraphType
from graphgallery.data.preprocess import largest_connected_components, create_subgraph


def _check_and_convert(adj_matrix: Optional[SparseMatrix]=None, 
                       attr_matrix: Optional[ArrayLike2D]=None, 
                       labels: Optional[ArrayLike1D]=None, copy: bool=True):
    # Make sure that the dimensions of matrices / arrays all agree
    if adj_matrix is not None:
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr(
                copy=False).astype(np.float32, copy=copy)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)"
                             .format(type(adj_matrix)))

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
                "Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)".format(type(attr_matrix)))

        if adj_matrix is not None and attr_matrix.shape[0] != adj_matrix.shape[0]:
            raise ValueError(
                "Dimensions of the adjacency and attribute matrices don't agree!")

    if labels is not None:
        labels = np.array(labels, dtype=np.int32, copy=copy)
        if labels.ndim != 1:
            labels = labels.argmax(1)

    return adj_matrix, attr_matrix, labels


class Graph(BaseGraph):
    """Attributed labeled graph stored in sparse matrix form."""

    # By default, the attr_matrix is dense format, i.e., Numpy array
    _sparse_attr = False

    def __init__(self, adj_matrix: Optional[SparseMatrix]=None, 
                       attr_matrix: Optional[Union[SparseMatrix, ArrayLike2D]]=None, 
                       labels: Optional[ArrayLike1D]=None,
                       node_names: List[str]=None, 
                       attr_names: List[str]=None, 
                       class_names: List[str]=None, 
                       metadata: str=None, 
                       copy: bool=True):
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
        adj_matrix, attr_matrix, labels = _check_and_convert(
            adj_matrix, attr_matrix, labels, copy=copy)

        if node_names is not None and adj_matrix is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency matrix and the node names don't agree!")

        if attr_names is not None and attr_matrix is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError(
                    "Dimensions of the attribute matrix and the attribute names don't agree!")

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels

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

    def set_inputs(self, adj_matrix: Optional[SparseMatrix]=None, 
                       attr_matrix: Optional[Union[SparseMatrix, ArrayLike2D]]=None, 
                       labels: Optional[ArrayLike1D]=None,
                       node_names: List[str]=None, 
                       attr_names: List[str]=None, 
                       class_names: List[str]=None, 
                       metadata: str=None, 
                       copy: bool=False):
        
        adj_matrix, attr_matrix, labels = _check_and_convert(adj_matrix, attr_matrix,
                                                             labels, copy=copy)

        if adj_matrix is not None:
            self.adj_matrix = adj_matrix

        if attr_matrix is not None:
            self.attr_matrix = attr_matrix

        if labels is not None:
            self.labels = labels

        if node_names is not None:
            if copy:
                self.node_names = copy_fn(node_names)
            else:
                self.node_names = node_names
        if attr_names is not None:
            if copy:
                self.attr_names = copy_fn(attr_names)
            else:
                self.attr_names = attr_names
        if class_names is not None:
            if copy:
                self.class_names = copy_fn(class_names)
            else:
                self.class_names = class_names
        if metadata is not None:
            if copy:
                self.metadata = copy_fn(metadata)
            else:
                self.metadata = metadata

    @property
    def sparse_attr(self) -> bool:
        return self._sparse_attr

    @sparse_attr.setter
    def sparse_attr(self, value):
        self._sparse_attr = value
        # clear LRU cache
        self.get_attr_matrix.cache_clear()

    @lru_cache(maxsize=1)
    def get_attr_matrix(self) -> Union[SparseMatrix, ArrayLike2D]:
        if self._attr_matrix is None:
            return None
        is_sparse = sp.isspmatrix(self._attr_matrix)
        if not self.sparse_attr and is_sparse:
            return self._attr_matrix.A
        elif self.sparse_attr and not is_sparse:
            return sp.csr_matrix(self._attr_matrix)
        return self._attr_matrix

    @property
    def adj_matrix(self) -> SparseMatrix:
        return self._adj_matrix

    @adj_matrix.setter
    def adj_matrix(self, x):
        self._adj_matrix = x

    @property
    def attr_matrix(self) -> Union[SparseMatrix, ArrayLike2D]:
        return self.get_attr_matrix()

    @attr_matrix.setter
    def attr_matrix(self, x):
        # clear LRU cache
        self.get_attr_matrix.cache_clear()
        self._attr_matrix = x

    @property
    def labels(self) -> ArrayLike1D:
        return self._labels

    @labels.setter
    def labels(self, x):
        self._labels = x

    @property
    def degrees(self) -> Union[Tuple[ArrayLike1D, ArrayLike1D], ArrayLike1D]:
        if not self.is_directed():
            return self.adj_matrix.sum(1).A1
        else:
            # in-degree and out-degree
            return self.adj_matrix.sum(0).A1, self.adj_matrix.sum(1).A1

    @property
    def n_nodes(self) -> int:
        """Get the number of nodes in the graph."""
        return self.adj_matrix.shape[0]

    @property
    def n_edges(self) -> int:
        """Get the number of edges in the graph.
        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

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
    def labels_onehot(self) -> ArrayLike2D:
        """Get the one-hot like labels of nodes."""
        labels = self.labels
        if labels is not None:
            return np.eye(self.n_classes)[labels].astype(labels.dtype)

    def neighbors(self, idx) -> ArrayLike1D:
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.
        """
        return self.adj_matrix[idx].indices

    def to_undirected(self) -> GraphType:
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

    def to_unweighted(self) -> GraphType:
        """Convert to an unweighted graph (set all edge weights to 1)."""
        G = self.copy()
        A = G.adj_matrix
        G.adj_matrix = sp.csr_matrix(
            (np.ones_like(A.data), A.indices, A.indptr), shape=A.shape)
        return G

    def eliminate_selfloops(self) -> GraphType:
        """Remove self-loops from the adjacency matrix."""
        G = self.copy()
        A = G.adj_matrix
        A = A - sp.diags(A.diagonal())
        A.eliminate_zeros()
        G.adj_matrix = A
        return G
    
    def eliminate_classes(self, threshold=0) -> GraphType:
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
            if count<=threshold:
                nodes_to_remove.extend(np.where(labels==_class)[0])
                removed += 1
            else:
                left.append(_class)
                
        if removed > 0:
            G = self.subgraph(nodes_to_remove=nodes_to_remove)
            mapping = dict(zip(left, range(self.n_classes-removed)))
            G.labels = np.asarray(list(map(lambda key: mapping[key], G.labels)), dtype=np.int32)      
            return G
        else:
            return self
    
    def add_selfloops(self, value=1.0) -> GraphType:
        """Set the diagonal."""
        G = self.eliminate_selfloops()
        A = G.adj_matrix
        A = A + sp.diags(A.diagonal() + value)
        A.eliminate_zeros()
        G.adj_matrix = A
        return G

    def standardize(self) -> GraphType:
        """Select the LCC of the unweighted/undirected/no-self-loop graph.
        All changes are done inplace.
        """
        G = self.to_unweighted().to_undirected().eliminate_selfloops()
        G = largest_connected_components(G, 1)
        return G

    def nxgraph(self, directed: bool=True) -> NxGraph:
        """Get the network graph from adj_matrix."""
        if directed:
            create_using = nx.DiGraph
        else:
            create_using = nx.Graph
        return nx.from_scipy_sparse_matrix(self.adj_matrix, create_using=create_using)
    
    def subgraph(self,  *, nodes_to_remove=None, nodes_to_keep=None) -> GraphType:
        return create_subgraph(self, nodes_to_remove=nodes_to_remove, nodes_to_keep=nodes_to_keep)

    def to_npz(self, filepath):
        filepath = save_graph_to_npz(filepath, self)
        print(f"save to {filepath}.")

    @staticmethod
    def from_npz(filepath) -> GraphType:
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
        if self.sparse_attr:
            return np.all(self.attr_matrix.data == 1)
        else:
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
        X_shape = self.adj_matrix.shape if self.adj_matrix is not None else (
            None, None)
        Y_shape = self.labels.shape if self.labels is not None else (None,)
        return f"{self.__class__.__name__}(adj_matrix{A_shape}, attr_matrix{X_shape}, labels{Y_shape})"


def load_dataset(data_path: str) -> GraphType:
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


def load_npz_to_graph(filename: str) -> GraphType:
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

        node_names = loader.get('node_names', None)
        attr_names = loader.get('attr_names', None)
        class_names = loader.get('class_names', None)
        metadata = loader.get('metadata', None)

    return Graph(adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata)


def save_graph_to_npz(filepath: str, graph: GraphType):
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
    elif isinstance(attr_matrix, np.ndarray):
        data_dict['attr_matrix'] = attr_matrix

    if sp.isspmatrix(labels):
        data_dict['labels_data'] = labels.data
        data_dict['labels_indices'] = labels.indices
        data_dict['labels_indptr'] = labels.indptr
        data_dict['labels_shape'] = labels.shape
    elif isinstance(labels, np.ndarray):
        data_dict['labels'] = labels

    if graph.node_names is not None:
        data_dict['node_names'] = graph.node_names

    if graph.attr_names is not None:
        data_dict['attr_names'] = graph.attr_names

    if graph.class_names is not None:
        data_dict['class_names'] = graph.class_names

    if graph.metadata is not None:
        data_dict['metadata'] = graph.metadata

    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'

    filepath = osp.abspath(osp.expanduser(osp.normpath(filepath)))
    np.savez(filepath, **data_dict)
    return filepath
