import numpy as np
import networkx as nx
import scipy.sparse as sp

from typing import Union, Optional, List, Tuple, Any
from ..data_type import is_listlike

NxGraph = Union[nx.Graph, nx.DiGraph]
Array1D = Union[List, np.ndarray]
Matrix2D = Union[List[List], np.ndarray]
ArrOrMatrix = Union[Array1D, Matrix2D]
AdjMatrix = Union[sp.csr_matrix, sp.csc_matrix]


def _check_adj_matrix(adj_matrix, copy=True):
    if sp.isspmatrix(adj_matrix):
        adj_matrix = adj_matrix.tocsr(
            copy=False).astype(np.float32, copy=copy)
    else:
        raise ValueError(f"Adjacency matrix must be in sparse format (got {type(adj_matrix)} instead).")

    assert adj_matrix.ndim == 2 and adj_matrix.shape[0] == adj_matrix.shape[1]
    return adj_matrix


def _check_attr_matrix(attr_matrix, copy=True):
    if sp.isspmatrix(attr_matrix):
        attr_matrix = attr_matrix.toarray().astype(np.float32, copy=False)
    elif isinstance(attr_matrix, np.ndarray):
        attr_matrix = attr_matrix.astype(np.float32, copy=copy)
    else:
        raise ValueError(
            f"Attribute matrix must be a scipy.sparse.spmatrix or a np.ndarray (got {type(attr_matrix)} instead).")

    assert attr_matrix.ndim == 2
    return attr_matrix


def _check_label_matrix(label_matrix, copy=True):
    if sp.isspmatrix(label_matrix):
        label_matrix = label_matrix.toarray().astype(np.int32, copy=False).squeeze()
    else:
        label_matrix = np.array(label_matrix, dtype=np.int32, copy=copy).squeeze()

    assert 0 < label_matrix.ndim <= 2
    # For one-hot like matrix, convert to 1D array
    if label_matrix.ndim == 2 and np.all(label_matrix.sum(1) == 1):
        label_matrix = label_matrix.argmax(1).astype(np.int32, copy=False)
    return label_matrix


EXCLUDE = {"metadata"}


def check_and_convert(key, value, multiple=False, copy=False) -> dict:
    if value is not None and key not in EXCLUDE:
        if "adj" in key:
            check_fn = _check_adj_matrix
        elif "attr" in key:
            check_fn = _check_attr_matrix
        else:
            check_fn = _check_label_matrix

        if multiple:
            if is_listlike(value):
                value = np.asarray([check_fn(v, copy=copy) for v in value])
            else:
                value = np.asarray([check_fn(value, copy=copy)])
        else:
            value = check_fn(value, copy=copy)

    return key, value


# def check(adj_matrix: Optional[AdjMatrix] = None,
#           node_attr: Optional[Matrix2D] = None,
#           node_labels: Optional[ArrOrMatrix] = None,
#           copy: bool = True):
#     # Make sure that the dimensions of matrices / arrays all agree
#     if adj_matrix is not None:
#         if sp.isspmatrix(adj_matrix):
#             adj_matrix = adj_matrix.tocsr(
#                 copy=False).astype(np.float32, copy=copy)
#         else:
#             raise ValueError(f"Adjacency matrix must be in sparse format (got {type(adj_matrix)} instead).")

#         assert adj_matrix.ndim == 2

#         if adj_matrix.shape[0] != adj_matrix.shape[1]:
#             raise ValueError("Dimensions of the adjacency matrix don't agree!")

#     if node_attr is not None:
#         if sp.isspmatrix(node_attr):
#             node_attr = node_attr.toarray().astype(np.float32, copy=False).squeeze()
#         elif isinstance(node_attr, np.ndarray):
#             node_attr = node_attr.astype(np.float32, copy=copy).squeeze()
#         else:
#             raise ValueError(
#                 f"Node attribute matrix must be a sp.spmatrix or a np.ndarray (got {type(node_attr)} instead).")

#         assert node_attr.ndim == 2
#     # elif adj_matrix is not None:
#     #     # TODO: is it necessary?
#     #     # Using identity matrix instead
#     #     node_attr = np.eye(adj_matrix.shape[0], dtype=np.float32)
#     if node_labels is not None:
#         if sp.isspmatrix(node_labels):
#             node_labels = node_labels.toarray().astype(np.int32, copy=False).squeeze()
#         else:
#             node_labels = np.array(node_labels, dtype=np.int32, copy=copy).squeeze()

#         assert 0 < node_labels.ndim <= 2
#         # For one-hot like matrix, convert to 1D array
#         if node_labels.ndim == 2 and np.all(node_labels.sum(1) == 1):
#             node_labels = node_labels.argmax(1).astype(np.int32, copy=False)

#     return adj_matrix, node_attr, node_labels
