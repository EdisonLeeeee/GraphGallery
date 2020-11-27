import numpy as np

from numba import njit
from typing import List

from graphgallery.typing import ArrayLike1D, SparseMatrix

__all__ = ['find_4o_nbrs']

@njit
def neighbors_mask(indices: ArrayLike1D, indptr: ArrayLike1D,
                   mask: ArrayLike1D, nbrs: ArrayLike1D, radius: int) -> ArrayLike1D:
    mask[nbrs] = True
    if radius <= 1:
        return mask
    else:
        for n in nbrs:
            next_nbrs = indices[indptr[n]:indptr[n + 1]]
            mask = neighbors_mask(indices, indptr, mask, next_nbrs, radius - 1)
        return mask


@njit
def _find_4o_nbrs(indices: ArrayLike1D, indptr: ArrayLike1D,
                  firstlevel: ArrayLike1D, radius: int = 4) -> List[ArrayLike1D]:
    N = len(indptr) - 1
    nodes = np.arange(N)
    for n in firstlevel:
        mask = np.asarray([False] * N)
        nbrs = indices[indptr[n]:indptr[n + 1]]
        mask = neighbors_mask(indices, indptr, mask, nbrs, radius)
        yield nodes[mask]


def find_4o_nbrs(adj_matrix: SparseMatrix,
                 candidates: ArrayLike1D = None,
                 radius: int = 4) -> List[ArrayLike1D]:
    if candidates is None:
        candidates = np.arange(adj_matrix.shape[0])
    return list(_find_4o_nbrs(adj_matrix.indices, adj_matrix.indptr, candidates, radius=radius))
