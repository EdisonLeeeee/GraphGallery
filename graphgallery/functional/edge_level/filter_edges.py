import warnings
import numpy as np
from .to_adj import asedge

__all__ = ["filter_singletons"]


def filter_singletons(edge, adj_matrix):
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

    edge = asedge(edge, shape="col_wise")  # shape [2, M]
    if edge.size == 0:
        warnings.warn("No edges found.", UserWarning)
        return edge

    degs = adj_matrix.sum(1).A1
    existing_edge = adj_matrix.tocsr(copy=False)[edge[0], edge[1]].A1

    if existing_edge.size > 0:
        edge_degrees = degs[edge] - 2 * existing_edge[None, :] + 1
    else:
        edge_degrees = degs[edge] + 1

    mask = np.logical_and(edge_degrees[0] > 0, edge_degrees[1] > 0)
    remained_edge = edge[:, mask]
    return remained_edge.T
