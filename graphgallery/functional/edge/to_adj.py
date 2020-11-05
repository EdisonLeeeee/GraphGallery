import numpy as np
import scipy.sparse as sp
from typing import Optional
from .shape import maybe_shape


__all__ = ['edge_transpose', 'edge_to_sparse_adj']


def edge_transpose(edge):
    """
    Transpose the transpose of an edge.

    Args:
        edge: (todo): write your description
    """
    edge = np.asarray(edge, dtype='int64')
    assert edge.ndim == 2
    M, N = edge.shape
    if not (M == 2 and N == 2) and N == 2:
        edge = edge.T
    return edge

# from ..transforms import Transform
# class EdgeToSparseAdj(Transform):
#     def __call__(self, edge_index: np.ndarray, edge_weight: Optional[np.ndarray] = None,
#                  shape: Optional[tuple] = None) -> sp.csr_matrix:
#         return sparse_adj_to_edge(edge_index=edge_index, edge_weight=edge_weight, shape=shape)

#     def __repr__(self):
#         return f"{self.__class__.__name__}()"


def edge_to_sparse_adj(edge: np.ndarray,
                       edge_weight: Optional[np.ndarray] = None,
                       shape: Optional[tuple] = None) -> sp.csr_matrix:
    """Convert (edge, edge_weight) representation to a Scipy sparse matrix

    Parameters
    ----------
    edge : np.ndarray
        edge index of sparse matrix, shape [2, M]
    edge_weight : Optional[np.ndarray], optional
        edge weight of sparse matrix, shape [M,], by default None
    shape : Optional[tuple], optional
        shape of sparse matrix, by default None

    Returns
    -------
    scipy.sparse.csr_matrix

    """
    edge = edge_transpose(edge)

    if edge_weight is None:
        edge_weight = np.ones(edge.shape[0], dtype=tk.floatx())

    if shape is None:
        shape = maybe_shape(edge)

    return sp.csr_matrix((edge_weight, edge), shape=shape)
