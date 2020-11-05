import numpy as np
import scipy.sparse as sp

from typing import Tuple
from ..transforms import Transform

__all__ = ['SparseAdjToEdge', 'sparse_adj_to_edge']

class SparseAdjToEdge(Transform):
    def __call__(self, adj_matrix: sp.csr_matrix) -> Tuple[np.ndarray]:
        """
        Return the adjacency matrix.

        Args:
            self: (todo): write your description
            adj_matrix: (array): write your description
            sp: (array): write your description
            csr_matrix: (array): write your description
        """
        return sparse_adj_to_edge(adj_matrix)

    def __repr__(self) -> str:
        """
        Return a repr representation of - repr representation.

        Args:
            self: (todo): write your description
        """
        return f"{self.__class__.__name__}()"


def sparse_adj_to_edge(adj_matrix: sp.csr_matrix) -> Tuple[np.ndarray]:
    """Convert a Scipy sparse matrix to (edge_index, edge_weight) representation
    """
    adj_matrix = adj_matrix.tocoo(copy=False)
    edge_index = np.asarray((adj_matrix.row, adj_matrix.col))
    edge_weight = adj_matrix.data.copy()

    return edge_index, edge_weight
