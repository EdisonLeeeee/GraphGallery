import warnings
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from .edge_transform import asedge

__all__ = ["singleton_filter", "connected_filter"]


def singleton_filter(edge, adj_matrix):
    """
    Filter edges that, if removed, would turn one or more nodes 
    into singleton nodes.

    Parameters
    ----------
    edge: np.array, shape [M, 2] or [2. M], where M is the number of input edges.
    adj_matrix: sp.sparse_matrix, shape [num_nodes, num_nodes]
        The input adjacency matrix.

    Returns
    -------
    np.array, shape [M, 2], 
        the edges that removed will not generate singleton nodes.
    """

    edge = asedge(edge, shape="row_wise")  # shape [M, 2]
    if edge.size == 0:
        warnings.warn("No edges found.", UserWarning)
        return edge

    degs = adj_matrix.sum(1).A1
    existing_edge = adj_matrix.tocsr(copy=False)[edge[:, 0], edge[:, 1]].A1

    if existing_edge.size > 0:
        edge_degrees = degs[edge] - 2 * existing_edge[:, None] + 1
    else:
        edge_degrees = degs[edge] + 1

    mask = np.logical_and(edge_degrees[:, 0] > 0, edge_degrees[:, 1] > 0)
    return edge[mask]


def connected_filter(edge, adj_matrix, score=None):
    edge = asedge(edge, shape="row_wise")  # shape [M,2]

    if edge.size == 0:
        warnings.warn("No edges found.", UserWarning)
        return edge

    if score is None:
        score = np.ones(edge.shape[0])
    else:
        assert edge.shape[0] == score.shape[0]

    edge_sym = np.vstack([edge, edge[:, [1, 0]]])
    score_sym = np.hstack([score, score])

    csgraph = adj_matrix.tolil(copy=True)
    csgraph[edge_sym[:, 0], edge_sym[:, 1]] = 1 - score_sym
    csgraph = csgraph.tocsr()
    mst = minimum_spanning_tree(csgraph)

    row, col = edge.T
    mask = np.logical_and(mst[(row, col)].A1 == 0, mst[(col, row)].A1 == 0)
    return edge[mask]
