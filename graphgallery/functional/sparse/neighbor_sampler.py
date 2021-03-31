import numpy as np
import scipy.sparse as sp

from numba import njit
from ..base_transforms import SparseTransform
from ..transform import Transform
from ..sparse import add_selfloops, eliminate_selfloops


@Transform.register()
class NeighborSampler(SparseTransform):

    def __init__(self, max_degree: int = 25,
                 selfloop: bool = True):
        super().__init__()
        self.collect(locals())

    def __call__(self, adj_matrix: sp.csr_matrix):
        return neighbor_sampler(adj_matrix, max_degree=self.max_degree,
                                selfloop=self.selfloop)


@njit
def sample(indices, indptr, max_degree=25):
    N = len(indptr) - 1
    M = N * np.ones((N + 1, max_degree), dtype=np.int32)
    for n in range(N):
        neighbors = indices[indptr[n]:indptr[n + 1]]
        size = neighbors.size

        if size > max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)
        elif size < max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=True)
        M[n] = neighbors
    return M


def neighbor_sampler(adj_matrix: sp.csr_matrix, max_degree: int = 25,
                     selfloop: bool = True):
    if selfloop:
        adj_matrix = add_selfloops(adj_matrix)
    else:
        adj_matrix = eliminate_selfloops(adj_matrix)

    M = sample(adj_matrix.indices, adj_matrix.indptr, max_degree=max_degree)
    np.random.shuffle(M.T)
    return M
