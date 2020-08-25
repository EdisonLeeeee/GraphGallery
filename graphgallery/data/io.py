import os
import errno
import numpy as np
import os.path as osp
import scipy.sparse as sp

from tensorflow.keras.utils import get_file
from graphgallery.data.preprocess import largest_connected_components


class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form."""

    def __init__(self, adj_matrix, attr_matrix=None, labels=None,
                 node_names=None, attr_names=None, class_names=None, metadata=None):
        """Create an (attributed and labeled) graph.

        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [n_nodes, n_nodes]
            Adjacency matrix in CSR format.
        attr_matrix : sp.csr_matrix or np.ndarray, shape [n_nodes, n_attrs], optional
            Attribute matrix in CSR or numpy format.
        labels : np.ndarray, shape [n_nodes], optional
            Array, where each entry represents respective node's label(s).
        node_names : np.ndarray, shape [n_nodes], optional
            Names of nodes (as strings).
        attr_names : np.ndarray, shape [n_attrs]
            Names of the attributes (as strings).
        class_names : np.ndarray, shape [n_classes], optional
            Names of the class labels (as strings).
        metadata : object
            Additional metadata such as text.
        """
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)"
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)"
                                 .format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree")

        if labels is not None:
            labels = np.array(labels, dtype=np.int32)
            if labels.ndim != 1:
                labels = labels.argmax(1)
            # if labels.shape[0] != adj_matrix.shape[0]:
                # raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree")

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

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
    def n_classes(self):
        """Get the number of classes labels of the nodes."""
        if self.labels is not None:
            return self.labels.max()+1
        else:
            raise ValueError("The node labels are not specified.")

    @property
    def n_attributes(self):
        """Get the number of attribute dimensions of the nodes."""
        if self.attr_matrix is not None:
            return self.attr_matrix.shape[1]
        else:
            raise ValueError("The node attribute matrix is not specified.")

    def get_neighbors(self, idx):
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.
        """
        return self.adj_matrix[idx].indices

    def to_dense_attrs(self):
        """Convert to dense attribute matrix (convert attribute matrix to Numpy array)."""
        attr_matrix = self.attr_matrix
        if sp.isspmatrix(attr_matrix):
            self.attr_matrix = attr_matrix.A
        else:
            pass
        return self

    def to_undirected(self):
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            A = self.adj_matrix
            A = A.maximum(A.T)
            self.adj_matrix = A
        return self

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    def eliminate_self_loops(self):
        """Remove self-loops from the adjacency matrix."""
        A = self.adj_matrix
        A -= sp.diags(A.diagonal())
        A.eliminate_zeros()
        self.adj_matrix = A
        return self

    def add_self_loops(self, value=1.0):
        """Set the diagonal."""
        self.eliminate_self_loops()
        A = self.adj_matrix
        A += sp.diags(A.diagonal()+value)
        if value == 0:
            A.eliminate_zeros()
        self.adj_matrix = A
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

    def to_npz(self, filepath):
        filepath = save_sparse_graph_to_npz(filepath, self)
        print(f"save to {filepath}.")

    @classmethod
    def from_npz(cls, filepath):
        return load_dataset(filepath)

    def copy(self):
        return SparseGraph(*self.unpack(), node_names=self.node_names,
                           attr_names=self.attr_names, class_names=self.class_names,
                           metadata=self.metadata)

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


def download_file(raw_paths, urls):

    last_except = None
    for file_name, url in zip(raw_paths, urls):
        try:
            get_file(file_name, origin=url)
        except Exception as e:
            last_except = e
            print(e)

    if last_except is not None:
        raise last_except


def files_exist(files):
    return len(files) != 0 and all([osp.exists(f) for f in files])


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def load_dataset(data_path):
    """Load a dataset.

    Parameters
    ----------
    name : str
        Name of the dataset to load.

    Returns
    -------
    sparse_graph : SparseGraph
        The requested dataset in sparse format.
    """
    data_path = osp.abspath(osp.expanduser(osp.normpath(data_path)))

    if not data_path.endswith('.npz'):
        data_path = data_path + '.npz'
    if osp.isfile(data_path):
        return load_npz_to_sparse_graph(data_path)
    else:
        raise ValueError(f"{data_path} doesn't exist.")


def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : SparseGraph
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
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return SparseGraph(adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata)


def save_sparse_graph_to_npz(filepath, sparse_graph):
    """Save a SparseGraph to a Numpy binary file.

    Parameters
    ----------
    filepath : str
        Name of the output file.
    sparse_graph : graphgalllery.data.SparseGraph
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
