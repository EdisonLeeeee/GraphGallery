import numpy as np
import scipy.sparse as sp

from ..transform import SparseTransform
from ..transform import Transform
from ..decorators import multiple

__all__ = ['SparseAdjToEdge', 'sparse_adj_to_edge']


@Transform.register()
class SparseAdjToEdge(SparseTransform):
    def __call__(self, adj_matrix: sp.csr_matrix):
        return sparse_adj_to_edge(adj_matrix)


@multiple()
def sparse_adj_to_edge(adj_matrix: sp.csr_matrix):
    """Convert a Scipy sparse matrix to (edge_index, edge_weight) representation"""
    adj_matrix = adj_matrix.tocoo(copy=False)
    edge_index = np.asarray((adj_matrix.row, adj_matrix.col))
    edge_weight = adj_matrix.data.copy()

    return edge_index, edge_weight
