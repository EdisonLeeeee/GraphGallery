import warnings
import numpy as np
import graphgallery as gg
import scipy.sparse as sp
from typing import Optional

__all__ = ['asedge', 'edge_to_sparse_adj']


def maybe_shape(edge):
    assert np.ndim(edge) == 2 and np.shape(edge)[0] == 2
    M = np.max(edge[0]) + 1
    N = np.max(edge[1]) + 1
    M = N = np.maximum(M, N)
    return M, N


def maybe_num_nodes(index: np.ndarray,
                    num_nodes: Optional[int] = None) -> int:

    return int(np.max(index)) + 1 if num_nodes is None else num_nodes


def asedge(edge: np.ndarray, shape="col_wise", symmetric=False, dtype=None):
    """make sure the array as edge like,
    shape [M, 2] or [2, M] with dtype 'dtype' (or 'int64')
    if ``symmetric=True``, it wiil have shape
    [2*M, 2] or [2, M*2].

    Parameters
    ----------
    edge : List, np.ndarray
        edge like list or array
    shape : str, optional
        row_wise: edge has shape [M, 2]
        col_wise: edge has shape [2, M]
        by default ``col_wise``
    symmetric: bool, optional
        if ``True``, the output edge will be 
        symmectric, i.e., 
        row_wise: edge has shape [2*M, 2]
        col_wise: edge has shape [2, M*2]
        by default ``False``
    dtype: string, optional
        data type for edges, if None, default to 'int64'

    Returns
    -------
    np.ndarray
        edge array
    """
    assert shape in ["row_wise", "col_wise"], shape
    assert isinstance(edge, (np.ndarray, list, tuple)), edge
    edge = np.asarray(edge, dtype=dtype or "int64")
    assert edge.ndim == 2 and 2 in edge.shape, edge.shape
    N, M = edge.shape
    if N == M == 2 and shape == "row_wise":
        # TODO: N=M=2 is confusing, we assume that edge was 'row_wise'
        warnings.warn(f"The shape of the edge is {N}x{M}."
                      f"we assume that {edge} was 'col_wise'")
        edge = edge.T
    elif (shape == "col_wise" and N != 2) or (shape == "row_wise" and M != 2):
        edge = edge.T

    if symmetric:
        if shape == "col_wise":
            edge = np.hstack([edge, edge[[1, 0]]])
        else:
            edge = np.vstack([edge, edge[:, [1, 0]]])

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

    edge = asedge(edge, shape="col_wise")

    if edge_weight is None:
        edge_weight = np.ones(edge.shape[1], dtype='float32')

    if shape is None:
        shape = maybe_shape(edge)
    return sp.csr_matrix((edge_weight, edge), shape=shape)
