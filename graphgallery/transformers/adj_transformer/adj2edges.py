import numpy as np
import scipy.sparse as sp
from graphgallery.transformers import Transformer


class SparseAdjToSparseEdges(Transformer):
    def __call__(self, adj_matrix):
        return sparse_adj_to_sparse_edges(adj_matrix)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    
    
def sparse_adj_to_sparse_edges(adj: sp.csr_matrix):
    """Convert a Scipy sparse matrix to (edge_index, edge_weight) representation

    edge_index: shape [M, 2]
    edge_weight: shape [M,]

    """
    adj = adj.tocoo()
    edge_index = np.stack([adj.row, adj.col], axis=1)
    edge_weight = adj.data

    return edge_index, edge_weight    