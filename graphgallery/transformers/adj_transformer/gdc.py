import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from scipy.linalg import expm

from graphgallery.transformers import normalize_adj
from graphgallery.transformers import Transformer
from graphgallery.utils.decorators import MultiInputs





class GDC(Transformer):
    def __init__(self, alpha: float=0.3, t: float=None, eps: float=None, k: int=128, which: str='PPR'):
        super().__init__()
        self.alpha = alpha
        self.t = t
        self.eps = eps
        self.k = k
        self.which = which
        
    def __call__(self, adj_matrix):
        return gdc(adj_matrix, alpha=self.alpha, t=self.t, eps=self.eps, k=self.k, which=self.which)

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, t={self.t}, eps={self.eps}, k={self.k}, which={self.which})"
    
    
@MultiInputs()
def gdc(adj_matrix: sp.csr_matrix, alpha: float=0.3, t: float=None, eps: float=None, k: int=128, which: str='PPR') -> sp.csr_matrix:

    if not (eps or k):
        raise RuntimeError('Either `eps` or `k` should be specified!')
    if eps and k:
        raise RuntimeError('Only one of `eps` and `k` should be specified!')

    N = adj_matrix.shape[0]

    # Symmetric transition matrix
    T_sym = normalize_adj(adj_matrix)

    if which == 'PPR':
        # PPR-based diffusion
        assert alpha, '`alpha` should be specified for PPR-based diffusion.'
        S = alpha * sp.linalg.inv((sp.eye(N) - (1 - alpha) * T_sym).tocsc())
    elif which == 'Heat':
        assert t, '`t` should be specified for Heat-based diffusion.'
        S = -t * (sp.eye(N) - T_sym)
        S = expm(S.toarray())
        S = sp.csr_matrix(S)
    else:
        raise ValueError(f'Invalid argument of `{which}`.')

    if eps:
        # Sparsify using threshold epsilon
        S = clip_matrix(S, threshold=eps)
    else:
        # Row-wise select top-k values
        S = top_k_matrix(S, k=k)

    # Column-normalized transition matrix on graph S_tilde
    T_S = normalize(S, norm='l1', axis=0)

    return T_S.tocsr(copy=False)

def clip_matrix(matrix, threshold: float) -> sp.csr_matrix:
    '''Sparsify using threshold epsilon'''
    assert sp.isspmatrix(matrix), 'Input matrix should be sparse matrix with format scipy.sparse.*_matrix.'
    matrix = matrix.tocsr()
    thres = np.vectorize(lambda x: x if x >= threshold else 0.)
    matrix.data = thres(matrix.data)
    matrix.eliminate_zeros()
    return matrix


def top_k_matrix(matrix, k: int) -> sp.csr_matrix:
    '''Row-wise select top-k values'''
    assert sp.isspmatrix(matrix), 'Input matrix should be sparse matrix with format scipy.sparse.*_matrix.'
    matrix = matrix.tolil()
    data = matrix.data
    for row in range(matrix.shape[0]):
        t = np.asarray(data[row])
        t[np.argsort(-t)[k:]] = 0.
        data[row] = t.tolist()
    matrix = matrix.tocsr()
    matrix.eliminate_zeros()
    return matrix