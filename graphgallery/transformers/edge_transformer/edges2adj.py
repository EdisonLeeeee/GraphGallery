import numpy as np
import scipy.sparse as sp
from graphgallery.transformers import Transformer


class SparseEdgesToSparseAdj(Transformer):
    def __call__(self, edge_index: np.ndarray, edge_weight: np.ndarray=None, shape=None) -> sp.csr_matrix:
        return sparse_adj_to_sparse_edges(edge_index=edge_index, edge_weight=edge_weight, shape=shape)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
def sparse_edges_to_sparse_adj(edge_index: np.ndarray, edge_weight: np.ndarray=None, shape=None) -> sp.csr_matrix:
    """Convert (edge_index, edge_weight) representation to a Scipy sparse matrix

    edge_index: shape [M, 2]
    edge_weight: shape [M,]

    """
    edges_shape = edge_index.shape
    assert np.ndim(edges_shape) == 2 and edges_shape[1] == 2
    
    if shape is None:
        N = np.max(edge_index) + 1
        shape = (N, N)
        
    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[0], dtype=floatx())        
        
    edge_index = edge_index.astype('int64', copy=False)
    adj = sp.csr_matrix(
        (edge_weight, (edge_index[:, 0], edge_index[:, 1])), shape=shape)
    return adj