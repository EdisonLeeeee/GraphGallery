import numpy as np
import scipy.sparse as sp

import numba
from numba import njit
from ..transform import SparseTransform
from ..transform import Transform
from ..sparse import add_self_loop, remove_self_loop


@Transform.register()
class ToNeighborMatrix(SparseTransform):

    def __init__(self, max_degree: int = 25,
                 self_loop: bool = True,
                 add_dummy: bool = True):
        super().__init__()
        self.collect(locals())

    def __call__(self, adj_matrix: sp.csr_matrix):
        return to_neighbor_matrix(adj_matrix, max_degree=self.max_degree,
                                  self_loop=self.self_loop, add_dummy=self.add_dummy)


@njit
def sample(indices, indptr, max_degree=25, add_dummy=True):
    N = len(indptr) - 1
    if add_dummy:
        M = numba.int32(N) + np.zeros((N + 1, max_degree), dtype=np.int32)
    else:
        M = np.zeros((N, max_degree), dtype=np.int32)

    for n in range(N):
        neighbors = indices[indptr[n]:indptr[n + 1]]
        size = neighbors.size

        if size > max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)
        elif size < max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=True)
        M[n] = neighbors
    return M


def to_neighbor_matrix(adj_matrix: sp.csr_matrix, max_degree: int = 25,
                       self_loop: bool = True, add_dummy=True):
    if self_loop:
        adj_matrix = add_self_loop(adj_matrix)
    else:
        adj_matrix = remove_self_loop(adj_matrix)

    M = sample(adj_matrix.indices, adj_matrix.indptr, max_degree=max_degree, add_dummy=add_dummy)
    np.random.shuffle(M.T)
    return M
