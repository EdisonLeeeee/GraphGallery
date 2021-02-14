import numpy as np
import scipy.sparse as sp
from .normalize_adj import normalize_adj
from ..base_transforms import SparseTransform
from ..decorators import multiple
from ..transform import Transform


@Transform.register()
class PPR(SparseTransform):
    def __init__(self,
                 alpha: float = 0.1):
        super().__init__()
        self.collect(locals())

    def __call__(self, adj_matrix):
        return ppr(adj_matrix,
                   alpha=self.alpha)


@multiple()
def ppr(adj_matrix: sp.csr_matrix, alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    M = normalize_adj(adj_matrix)
    A_inner = sp.eye(num_nodes, format='csr') - (1 - alpha) * M
    return alpha * np.linalg.inv(A_inner.toarray())
