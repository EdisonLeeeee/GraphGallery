import numpy as np
import scipy.sparse as sp

_SPARSE_THRESHOLD = 0.5


def sparse_collate(key, val):
    if isinstance(val, np.ndarray) and val.ndim == 2:
        # one-hot like matrix stored with 1D array
        if "labels" in key and np.all(val.sum(1) == 1):
            val = val.argmax(1)
        else:
            shape = val.shape
            # identity matrix, do not store in files
            if shape[0] == shape[1] and np.diagonal(val).sum() == shape[0]:
                val = None
            else:
                sparsity = (val == 0).sum() / val.size
                # if sparse enough, store as sparse matrix
                if sparsity > _SPARSE_THRESHOLD:
                    val = sp.csr_matrix(val)

    return key, val
