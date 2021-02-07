import numpy as np
import scipy.sparse as sp

from ..transforms import BaseTransform
from ..get_transform import Transform
from graphgallery import intx


@Transform.register()
class NeighborSampler(BaseTransform):

    def __init__(self, max_degree: int = 25,
                 selfloop: bool = False):
        super().__init__()
        self.max_degree = max_degree
        self.selfloop = selfloop

    def __call__(self, adj_matrix: sp.csr_matrix):
        return neighbor_sampler(adj_matrix, max_degree=self.max_degree,
                                selfloop=self.selfloop)

    def extra_repr(self):
        return f"max_degree={self.max_degree}, selfloop={self.selfloop}"


def neighbor_sampler(adj_matrix: sp.csr_matrix, max_degree: int = 25,
                     selfloop: bool = False):
    adj_matrix = adj_matrix.tocsr(copy=False)
    N = adj_matrix.shape[0]
    neighbors_matrix = N * np.ones((N + 1, max_degree), dtype=intx())
    for nodeid in range(N):
        neighbors = adj_matrix[nodeid].indices

#         if not selfloop:
#             neighbors = np.setdiff1d(neighbors, [nodeid])
#         else:
#             neighbors = np.intersect1d(neighbors, [nodeid])

        size = neighbors.size
        if size == 0:
            continue

        if size > max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)
        elif size < max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=True)

        neighbors_matrix[nodeid] = neighbors

    np.random.shuffle(neighbors_matrix.T)
    return neighbors_matrix
