import scipy.sparse as sp
import numpy as np

from graphgallery.transforms import Transform
from graphgallery.transforms import normalize_adj
from graphgallery.utils.decorators import MultiInputs


class ChebyBasis(Transform):
    def __init__(self, order=2, rate=-0.5):
        super().__init__()
        self.order = order
        self.rate = rate

    def __call__(self, adj_matrix):
        return cheby_basis(adj_matrix, order=self.order, rate=self.rate)

    def __repr__(self):
        return f"{self.__class__.__name__}(order={self.order}, rate={self.rate})"

@MultiInputs()
def cheby_basis(adj_matrix, order=2, rate=-0.5):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""

    assert order >= 2
    adj_normalized = normalize_adj(adj_matrix, rate=rate, fill_weight=1.0)
    I = sp.eye(adj_matrix.shape[0], dtype=adj_matrix.dtype).tocsr()
    laplacian = I - adj_normalized
    largest_eigval = sp.linalg.eigsh(
        laplacian, 1, which='LM', return_eigenvectors=False)[0]
    scaled_laplacian = (2. / largest_eigval) * laplacian - I

    t_k = []
    t_k.append(I)
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        return 2 * scaled_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, order + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return t_k
