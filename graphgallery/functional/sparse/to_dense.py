import numpy as np
import scipy.sparse as sp

from ..base_transforms import SparseTransform
from ..transform import Transform
from ..decorators import multiple

__all__ = ['ToDense', 'to_dense']


@Transform.register()
class ToDense(SparseTransform):
    def __call__(self, adj_matrix: sp.csr_matrix):
        return to_dense(adj_matrix)


@multiple()
def to_dense(adj_matrix: sp.csr_matrix):
    """Convert a Scipy sparse matrix to (edge_index, edge_weight) representation"""
    if sp.isspmatrix(adj_matrix):
        return adj_matrix.A
    elif isinstance(adj_matrix, np.ndarray):
        return adj_matrix
    else:
        raise TypeError(type(adj_matrix))