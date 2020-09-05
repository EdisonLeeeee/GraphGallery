import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from scipy.linalg import expm

from graphgallery.utils.data_utils import normalize_adj


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


def GDC(adj: sp.csr_matrix, alpha: float = None, t: float = None, eps: float = None, k: int = None, which: str = 'PPR') -> sp.csr_matrix:

    if not (eps or k):
        raise RuntimeError('Either `eps` or `k` should be specified!')
    if eps and k:
        raise RuntimeError('Only one of `eps` and `k` should be specified!')

    N = adj.shape[0]

    # Symmetric transition matrix
    T_sym = normalize_adj(adj)

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

    return T_S.tocsr()
