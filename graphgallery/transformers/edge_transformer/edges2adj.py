import numpy as np
import scipy.sparse as sp
from graphgallery.transformers import Transformer
from graphgallery.transformers import edge_transpose


class SparseEdgesToSparseAdj(Transformer):
    def __call__(self, edge_index: np.ndarray, edge_weight: np.ndarray=None, shape=None) -> sp.csr_matrix:
        return sparse_adj_to_sparse_edges(edge_index=edge_index, edge_weight=edge_weight, shape=shape)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
def sparse_edges_to_sparse_adj(edge_index: np.ndarray, edge_weight: np.ndarray=None, shape=None) -> sp.csr_matrix:
    """Convert (edge_index, edge_weight) representation to a Scipy sparse matrix

    edge_index: shape [M, 2] or [2, M] -> [2, M]
    edge_weight: shape [M,]

    """
    edge_index = edge_transpose(edge_index)
    
    if shape is None:
        N = np.max(edge_index) + 1
        shape = (N, N)
        
    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[0], dtype=floatx())        
        
    edge_index = edge_index.astype('int64', copy=False)
    adj = sp.csr_matrix((edge_weight, edge_index), shape=shape)
    return adj