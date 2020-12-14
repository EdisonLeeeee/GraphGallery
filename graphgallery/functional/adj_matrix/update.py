import warnings
import numpy as np
import scipy.sparse as sp
from ..edge_level import asedge
from ..attr_matrix import flip_attr

__all__ = ["flip_adj", "add_edge", "remove_edge"]


def flip_adj(adj_matrix, flips, symmetric=True):
    if flips is None or len(flips) == 0:
        warnings.warn(
            "There is NO structure flips, the adjacency matrix stays the same.",
            RuntimeWarning,
        )
        return adj_matrix.copy()

    flips = asedge(flips, symmetric=symmetric)

    if isinstance(adj_matrix, np.ndarray):
        return flip_attr(adj_matrix, flips)
    elif not sp.isspmatrix(adj_matrix):
        raise ValueError(f"adj_matrix must be a Scipy sparse matrix or Numpy array, but got {type(adj_matrix)}.")

    # TODO: for adj_matrix is weighted？
    data = adj_matrix[tuple(flips)].A
    data[data > 0.] = 1.
    data[data < 0.] = 0.

    adj_matrix = adj_matrix.tolil(copy=True)
    adj_matrix[tuple(flips)] = 1. - data
    adj_matrix = adj_matrix.tocsr(copy=False)
    adj_matrix.eliminate_zeros()

    return adj_matrix


def add_edge(adj_matrix, edges, symmetric=True):
    if edges is None or len(edges) == 0:
        warnings.warn(
            "There is NO structure edges, the adjacency matrix stays the same.",
            RuntimeWarning,
        )
        return adj_matrix.copy()

    edges = asedge(edges, symmetric=symmetric)

    if np.any(adj_matrix[tuple(edges)]):
        warnings.warn(
            "Some edges already exist, and adding them may cause an error.",
            RuntimeWarning,
        )

    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = adj_matrix.copy()
        adj_matrix[edges] += 1.
        return adj_matrix
    elif not sp.isspmatrix(adj_matrix):
        raise ValueError(f"adj_matrix must be a Scipy sparse matrix or Numpy array, but got {type(adj_matrix)}.")

    datas = np.ones(edges.shape[1], dtype=adj_matrix.dtype)

    adj_matrix = adj_matrix.tocoo(copy=True)
    edges = np.hstack([edges, [adj_matrix.row, adj_matrix.col]])
    datas = np.hstack([adj_matrix.data, datas])
    adj_matrix = sp.csr_matrix((datas, tuple(edges)), shape=adj_matrix.shape)
    adj_matrix.eliminate_zeros()
    return adj_matrix


def remove_edge(adj_matrix, edges, symmetric=True):
    if edges is None or len(edges) == 0:
        warnings.warn(
            "There is NO structure edges, the adjacency matrix stays the same.",
            RuntimeWarning,
        )
        return adj_matrix.copy()

    edges = asedge(edges, symmetric=symmetric)

    if not np.all(adj_matrix[tuple(edges)]):
        warnings.warn(
            "Some edges that don't exist, and removing them may cause an error.",
            RuntimeWarning,
        )

    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = adj_matrix.copy()
        adj_matrix[edges] -= 1.
        adj_matrix[adj_matrix <= 0] = 0
        return adj_matrix
    elif not sp.isspmatrix(adj_matrix):
        raise ValueError(f"adj_matrix must be a Scipy sparse matrix or Numpy array, but got {type(adj_matrix)}.")

    datas = -np.ones(edges.shape[1], dtype=adj_matrix.dtype)

    adj_matrix = adj_matrix.tocoo(copy=True)
    edges = np.hstack([edges, [adj_matrix.row, adj_matrix.col]])
    datas = np.hstack([adj_matrix.data, datas])
    adj_matrix = sp.csr_matrix((datas, tuple(edges)), shape=adj_matrix.shape)
    adj_matrix[adj_matrix < 0] = 0.
    adj_matrix.eliminate_zeros()
    return adj_matrix
