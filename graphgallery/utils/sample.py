from numba import njit
import numpy as np


@njit
def neighbors_mask(indices, indptr, mask, nbrs, radius):
    mask[nbrs] = True
    if radius <= 1:
        return mask
    else:
        for n in nbrs:
            next_nbrs = indices[indptr[n]:indptr[n + 1]]
            mask = neighbors_mask(indices, indptr, mask, next_nbrs, radius - 1)
        return mask


@njit
def _find_4o_nbrs(indices, indptr, firstlevel, radius=4):
    N = len(indptr) - 1
    nodes = np.arange(N)
    for n in firstlevel:
        mask = np.asarray([False] * N)
        nbrs = indices[indptr[n]:indptr[n + 1]]
        mask = neighbors_mask(indices, indptr, mask, nbrs, radius)
        yield nodes[mask]


def find_4o_nbrs(adj_matrix, candidates=None, radius=4):
    if candidates is None:
        candidates = np.arange(adj_matrix.shape[0])
    return list(_find_4o_nbrs(adj_matrix.indices, adj_matrix.indptr, candidates, radius=radius))
