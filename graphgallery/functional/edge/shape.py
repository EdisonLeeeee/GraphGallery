import numpy as np

__all__ = ['maybe_shape']

def maybe_shape(edge):
    M = np.max(edge[0]) + 1
    N = np.max(edge[1]) + 1
    return M, N