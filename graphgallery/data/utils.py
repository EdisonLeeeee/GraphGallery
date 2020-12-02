import numpy as np
import networkx as nx
import scipy.sparse as sp

from typing import Union, Optional, List, Tuple, Any, Callable
from ..data_type import is_multiobjects

__all__ = ["get_num_nodes", "get_num_edges", "get_num_graphs",
           "get_num_node_attrs", "get_num_node_classes", "get_degree",
           "is_directed", "is_singleton", "is_selfloops",
           "is_binary", "is_weighted"
           ]


def get_degree(adj_matrices):
    assert adj_matrices is not None

    if is_multiobjects(adj_matrices):
        return tuple(get_degree(adj_matrix) for adj_matrix in adj_matrices)

    if not is_directed(adj_matrices):
        return adj_matrices.sum(1).A1
    else:
        # in-degree and out-degree
        return adj_matrices.sum(0).A1, adj_matrices.sum(1).A1


def get_num_nodes(adj_matrices, fn=sum):
    if adj_matrices is None:
        return 0

    if is_multiobjects(adj_matrices):
        return fn(get_num_nodes(adj_matrix) for adj_matrix in adj_matrices)
        # # NOTE: please make sure all the graph are the same!!
        # return max(get_num_nodes(adj_matrix) for adj_matrix in adj_matrices)

    return adj_matrices.shape[0]


def get_num_graphs(adj_matrices, fn=None):
    if adj_matrices is None:
        return 0

    if is_multiobjects(adj_matrices):
        # return sum(get_num_graphs(adj_matrix) for adj_matrix in adj_matrices)
        return len(adj_matrices)

    return 1


def get_num_edges(adj_matrices, is_directed=False, fn=sum):
    if adj_matrices is None:
        return 0

    if is_multiobjects(adj_matrices):
        return fn(get_num_edges(adj_matrix) for adj_matrix in adj_matrices)

    if is_directed:
        return int(adj_matrices.nnz)
    else:
        A = adj_matrices
        num_diag = (A.diagonal() != 0).sum()
        return int((A.nnz - num_diag) / 2) + int(num_diag)


def get_num_node_attrs(node_attrs, fn=max):
    if node_attrs is None:
        return 0

    if is_multiobjects(node_attrs):
        return fn(get_num_node_attrs(node_attr) for node_attr in node_attrs)

    return node_attrs.shape[1]


def get_num_node_classes(node_labels, fn=max):
    if node_labels is None:
        return 0

    if is_multiobjects(node_labels):
        return fn(
            get_num_node_classes(node_label) for node_label in node_labels)

    if node_labels.ndim == 1:
        return node_labels.max() + 1
    else:
        return node_labels.shape[1]


def is_directed(A) -> bool:
    assert A is not None
    if is_multiobjects(A):
        return any(is_directed(adj) for adj in A)
    return (A != A.T).sum() != 0


def is_singleton(A) -> bool:
    assert A is not None
    if is_multiobjects(A):
        return any(is_singleton(adj) for adj in A)
    out_deg = A.sum(1).A1
    in_deg = A.sum(0).A1
    return np.any(np.logical_and(in_deg == 0, out_deg == 0))


def is_selfloops(A) -> bool:
    assert A is not None
    if is_multiobjects(A):
        return any(is_selfloops(adj) for adj in A)
    return A.diagonal().sum() != 0


def is_binary(self) -> bool:
    assert A is not None
    if is_multiobjects(A):
        return all(is_binary(adj) for adj in A)
    return np.all(np.unique(A) == (0, 1))


def is_weighted(self) -> bool:
    assert A is not None
    if is_multiobjects(A):
        return any(is_weighted(adj) for adj in A)
    return np.any(A.data != 1)
