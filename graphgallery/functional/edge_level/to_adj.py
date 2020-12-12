import numpy as np
import scipy.sparse as sp
import graphgallery as gg
from typing import Optional
from .shape import maybe_shape


__all__ = ['asedge', 'edge_to_sparse_adj']


def asedge(edge, shape="col_wise"):
    """make sure the array as edge like,
    shape [M, 2] or [2, M] with dtype int64

    Parameters
    ----------
    edge : List, np.ndarray
        edge like list or array
    shape : str, optional
        row_wise: edge has shape [M, 2]
        col_wise: edge has shape [2, M]
        by default ``col_wise``

    Returns
    -------
    np.ndarray
        edge array
    """
    assert shape in ["row_wise", "col_wise"], shape
    edge = np.asarray(edge, dtype='int64')
    assert edge.ndim == 2 and 2 in edge.shape, edge.shape
    N, M = edge.shape
    if not (N == M == 2) and ((shape == "col_wise" and N != 2)
                              or (shape == "row_wise" and M != 2)):
        edge = edge.T

    return edge


def edge_to_sparse_adj(edge: np.ndarray,
                       edge_weight: Optional[np.ndarray] = None,
                       shape: Optional[tuple] = None) -> sp.csr_matrix:
    """Convert (edge, edge_weight) representation to a Scipy sparse matrix

    Parameters
    ----------
    edge : list or np.ndarray
        edge index of sparse matrix, shape [2, M]
    edge_weight : Optional[np.ndarray], optional
        edge weight of sparse matrix, shape [M,], by default None
    shape : Optional[tuple], optional
        shape of sparse matrix, by default None

    Returns
    -------
    scipy.sparse.csr_matrix

    """

    edge = asedge(edge)

    if edge_weight is None:
        edge_weight = np.ones(edge.shape[1], dtype=gg.floatx())

    if shape is None:
        shape = maybe_shape(edge)
    return sp.csr_matrix((edge_weight, edge), shape=shape)
