import scipy.sparse as sp
from sklearn.preprocessing import normalize
from scipy.linalg import expm

from .normalize_adj import normalize_adj
from ..transforms import BaseTransform
from ..decorators import multiple
from ..get_transform import Transform
from .topk import sparse_topk
from .clip import sparse_clip


@Transform.register()
class GDC(BaseTransform):
    def __init__(self,
                 alpha: float = 0.3,
                 t: float = None,
                 eps: float = None,
                 K: int = 128,
                 which: str = 'PPR'):
        super().__init__()
        self.alpha = alpha
        self.t = t
        self.eps = eps
        self.K = K
        self.which = which

    def __call__(self, adj_matrix):
        return gdc(adj_matrix,
                   alpha=self.alpha,
                   t=self.t,
                   eps=self.eps,
                   K=self.K,
                   which=self.which)

    def extra_repr(self):
        return f"alpha={self.alpha}, t={self.t}, eps={self.eps}, K={self.K}, which={self.which}"


@multiple()
def gdc(adj_matrix: sp.csr_matrix,
        alpha: float = 0.3,
        t: float = None,
        eps: float = None,
        K: int = 128,
        which: str = 'PPR') -> sp.csr_matrix:

    if not (eps or K):
        raise RuntimeError('Either `eps` or `K` should be specified!')
    if eps and K:
        raise RuntimeError('Only one of `eps` and `K` should be specified!')

    N = adj_matrix.shape[0]

    # Symmetric transition matrix
    T_sym = normalize_adj(adj_matrix)

    if which == 'PPR':
        # PPR-based diffusion
        assert alpha, '`alpha` should be specified for PPR-based diffusion.'
        S = alpha * sp.linalg.inv((sp.eye(N, format='csr') - (1 - alpha) * T_sym).tocsc())
    elif which == 'Heat':
        assert t, '`t` should be specified for Heat-based diffusion.'
        S = -t * (sp.eye(N, format='csr') - T_sym)
        S = expm(S.toarray())
        S = sp.csr_matrix(S)
    else:
        raise ValueError(f'Invalid argument of `{which}`.')

    if eps:
        # Sparsify using threshold epsilon
        S = sparse_clip(S, threshold=eps)
    else:
        # Row-wise select top-K values
        S = sparse_topk(S, K=K)

    # Column-normalized transition matrix on graph S_tilde
    T_S = normalize(S, norm='l1', axis=0)

    return T_S.tocsr(copy=False)
