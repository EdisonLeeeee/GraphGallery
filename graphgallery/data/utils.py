import numpy as np
import scipy.sparse as sp
from graphgallery import functional as gf

from ..data_type import is_multiobjects

__all__ = ["get_num_nodes", "get_num_edges", "get_num_graphs",
           "get_num_node_attrs", "get_num_node_classes",
           ]

identity = lambda x: x

def get_num_nodes(adj_matrices, reduce=identity):
    if adj_matrices is None:
        return 0

    if is_multiobjects(adj_matrices):
        return reduce(get_num_nodes(adj_matrix) for adj_matrix in adj_matrices)
        # # NOTE: please make sure all the graph are the same!!
        # return max(get_num_nodes(adj_matrix) for adj_matrix in adj_matrices)

    return adj_matrices.shape[0]


def get_num_graphs(adj_matrices):
    if adj_matrices is None:
        return 0

    if is_multiobjects(adj_matrices):
        # return sum(get_num_graphs(adj_matrix) for adj_matrix in adj_matrices)
        return len(adj_matrices)

    return 1


def get_num_edges(adj_matrices, is_directed=False, reduce=identity):
    if adj_matrices is None:
        return 0

    if is_multiobjects(adj_matrices):
        return reduce(get_num_edges(adj_matrix) for adj_matrix in adj_matrices)

    if is_directed:
        return int(adj_matrices.nnz)
    else:
        A = adj_matrices
        num_diag = (A.diagonal() != 0).sum()
        return int((A.nnz - num_diag) / 2) + int(num_diag)


def get_num_node_attrs(node_attrs, reduce=identity):
    if node_attrs is None:
        return 0

    if is_multiobjects(node_attrs):
        return reduce(get_num_node_attrs(node_attr) for node_attr in node_attrs)

    return node_attrs.shape[1]


def get_num_node_classes(node_labels, reduce=identity):
    if node_labels is None:
        return 0

    if is_multiobjects(node_labels):
        return reduce(get_num_node_classes(node_label) for node_label in node_labels)

    if node_labels.ndim == 1:
        return node_labels.max() + 1
    else:
        return node_labels.shape[1]
