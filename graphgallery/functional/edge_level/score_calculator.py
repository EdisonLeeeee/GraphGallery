import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from .edge_transform import asedge

__all__ = ["jaccard_score", "cosine_score",
           "kld_score", "svd_score", "entropy_score"]


def jaccard_score(edge, adj_matrix, matrix):
    edge = asedge(edge, shape="row_wise")  # shape [M, 2]
    rows, cols = edge.T
    assert np.ndim(matrix) == 2
    A = matrix[rows]
    B = matrix[cols]
    intersection = np.count_nonzero(A * B, axis=1)
    J = jaccard_similarity(A, B)
    return J


def cosine_score(edge, adj_matrix, matrix):
    edge = asedge(edge, shape="row_wise")  # shape [M, 2]
    rows, cols = edge.T
    assert np.ndim(matrix) == 2
    A = matrix[rows]
    B = matrix[cols]
    inner_product = (A * B).sum(1)
    C = cosine_similarity(A, B)
    return C


def svd_score(edge, adj_matrix, matrix=None, k=50):
    edge = asedge(edge, shape="row_wise")  # shape [M, 2]
    rows, cols = edge.T
    assert np.ndim(matrix) == 2
    adj_matrix = adj_matrix.asfptype()
    U, S, V = sp.linalg.svds(adj_matrix, k=k)
    adj_matrix = (U * S) @ V
    return adj_matrix[rows, cols]


def kld_score(edge, adj_matrix, matrix):
    edge = asedge(edge, shape="row_wise")  # shape [M, 2]
    rows, cols = edge.T
    assert np.ndim(matrix) == 2
    A = tf.gather(matrix, rows)
    B = tf.gather(matrix, cols)
    D = kld_divergence(A, B)
    return -D.numpy()


def entropy_score(edge, adj_matrix, matrix):
    edge = asedge(edge, shape="row_wise")  # shape [M, 2]
    rows, cols = edge.T
    assert np.ndim(matrix) == 2
    num_nodes = adj_matrix.shape[0]
    node_en = np.zeros(num_nodes)
    deg = adj_matrix.sum(1).A1

    for node in range(num_nodes):
        neighbors = adj_matrix[node].indices
        node_en[node] = neighborhood_entropy(matrix[node], matrix[neighbors])

    def get_info(node, nbr):
        neighbors = adj_matrix[node].indices.tolist()
        neighbors.remove(nbr)
        info_nbr = matrix[neighbors]

        return info_nbr

    S = np.zeros(rows.shape[0])
    for i, (u, v) in enumerate(zip(rows, cols)):
        if deg[u] <= 1 or deg[v] <= 1:
            continue

        info_u = matrix[u]
        info_v = matrix[v]

        info_nbr_u = get_info(u, v)
        info_nbr_v = get_info(v, u)

        entropy_u = neighborhood_entropy(info_u, info_nbr_u)
        entropy_v = neighborhood_entropy(info_v, info_nbr_v)
        S[i] = (entropy_u + entropy_v) - (node_en[u] + node_en[v])

    return S
