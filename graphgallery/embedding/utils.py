import numpy as np
import scipy.sparse as sp

def normalized_laplacian_matrix(graph, r=-0.5):
    graph = graph + sp.eye(graph.shape[0], dtype=graph.dtype, format='csr')

    degree = graph.sum(1).A1
    degree_power = np.power(degree, r)
    graph = graph.tocoo(copy=False)
    graph.data *= degree_power[graph.row]
    graph.data *= degree_power[graph.col]
    graph = graph.tocsr(copy=False)
    return graph