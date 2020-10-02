import numpy as np

def edge_transpose(edge_index):
    edge_index = np.asarray(edge_index)
    assert edge_index.ndim == 2
    M, N = edge_index.shape
    if not (M==2 and N==2) and N==2:
        edge_index = edge_index.T
    return edge_index