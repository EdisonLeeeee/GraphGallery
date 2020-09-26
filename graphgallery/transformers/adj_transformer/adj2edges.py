import numpy as np
import scipy.sparse as sp
from graphgallery.transformers import Transformer


class SparseAdjToSparseEdges(Transformer):
    def __call__(self, adj_matrix):
        return sparse_adj_to_sparse_edges(adj_matrix)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    
    
def sparse_adj_to_sparse_edges(adj_matrix: sp.csr_matrix):
    """Convert a Scipy sparse matrix to (edge_index, edge_weight) representation

    edge_index: shape [2, M]
    edge_weight: shape [M,]

    """
    adj_matrix = adj_matrix.tocoo(copy=False)
    edge_index = np.asarray((adj_matrix.row, adj_matrix.col))
    edge_weight = adj_matrix.data.copy()

    return edge_index, edge_weight    