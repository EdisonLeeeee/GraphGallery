from numba import njit
import numpy as np

@njit
def mask_neighbors(indices, indptr, mask, nbrs, radius):
    mask[nbrs] = True
    if radius <= 1:
        return mask
    else:
        for n in nbrs:
            next_nbrs = indices[indptr[n]:indptr[n + 1]]
            mask = mask_neighbors(indices, indptr, mask, next_nbrs, radius-1)
        return mask
    
@njit
def find_4o_nbrs(indices, indptr, firstlevel, radius=4):
    N = len(indptr) - 1
    nodes = np.arange(N)
    for n in firstlevel:
        mask = np.asarray([False]*N)
        nbrs = indices[indptr[n]:indptr[n + 1]]
        mask = mask_neighbors(indices, indptr, mask, nbrs, radius)
        yield nodes[mask]
