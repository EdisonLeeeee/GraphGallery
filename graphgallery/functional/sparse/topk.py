import numba as nb
import numpy as np
import scipy.sparse as sp

__all__ = ["sparse_topk"]


@nb.njit
def row_topk_csr(data, indptr, K):
    N = len(indptr) - 1
    for i in nb.prange(N):
        d = data[indptr[i]:indptr[i + 1]]

        if d.size > K:
            bottom_inds = np.argsort(d)[:(d.size - K)]
            data[indptr[i]:indptr[i + 1]][bottom_inds] = 0.
    return data


def sparse_topk(adj_matrix, K, axis=1):
    assert axis == 1, "not implemented"
    data = row_topk_csr(adj_matrix.data.copy(), adj_matrix.indptr, K)
    adj_matrix = sp.csr_matrix((data, adj_matrix.indices.copy(), adj_matrix.indptr.copy()), shape=adj_matrix.shape)
    adj_matrix.eliminate_zeros()
    return adj_matrix
