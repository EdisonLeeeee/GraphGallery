import numpy as np

def augment_attr(attr_matrix, N, data=0.):
    M = np.zeros([N, attr_matrix.shape[1]], dtype=attr_matrix.dtype) + data
    augmented_attr = np.vstack([attr_matrix, M])
    return augmented_attr