import scipy.sparse as sp

from .normalize_adj import normalize_adj
from ..transforms import Transform
from ..decorators import multiple
from ..get_transform import Transformers


@Transformers.register()
class ChebyBasis(Transform):
    def __init__(self, K=2, rate=-0.5):
        super().__init__()
        self.K = K
        self.rate = rate

    def __call__(self, adj_matrix):
        return cheby_basis(adj_matrix, K=self.K, rate=self.rate)

    def extra_repr(self):
        return f"K={self.K}, rate={self.rate}"


@multiple()
def cheby_basis(adj_matrix, K=2, rate=-0.5):
    """Calculate Chebyshev polynomials up to K k. Return a list of sparse matrices (tuple representation)."""

    assert K >= 2, K
    adj_normalized = normalize_adj(adj_matrix, rate=rate, fill_weight=1.0)
    I = sp.eye(adj_matrix.shape[0], dtype=adj_matrix.dtype, format='csr')
    laplacian = I - adj_normalized
    largest_eigval = sp.linalg.eigsh(laplacian,
                                     1,
                                     which='LM',
                                     return_eigenvectors=False)[0]
    scaled_laplacian = (2. / largest_eigval) * laplacian - I

    t_k = []
    t_k.append(I)
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        return 2 * scaled_lap.dot(t_k_minus_one) - t_k_minus_two

    for _ in range(2, K + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return t_k
