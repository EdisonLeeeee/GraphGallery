import numpy as np


__all__ = ["sparse_clip"]


def sparse_clip(matrix, threshold: float):
    matrix = matrix.tocsr(copy=False).copy()
    thres = np.vectorize(lambda x: x if x >= threshold else 0.)
    matrix.data = thres(matrix.data)
    matrix.eliminate_zeros()
    return matrix
