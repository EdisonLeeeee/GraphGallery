import warnings
import numpy as np
import scipy.sparse as sp
from graphgallery import functional as gf

from ..data_type import is_multiobjects

__all__ = ['sparse_apply', 'check_and_convert', 'index_select']

_SPARSE_THRESHOLD = 0.5


def sparse_apply(key, val):
    if is_multiobjects(val):
        return key, tuple(sparse_apply(key, v)[1] for v in val)

    if isinstance(val, np.ndarray) and val.ndim == 2:
        # one-hot like matrix stored with 1D array
        if "label" in key and np.all(val.sum(1) == 1):
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


def index_select(key, value, index, escape=None):
    if (isinstance(value, np.ndarray) or sp.isspmatrix(value)) and \
            (escape is None or key not in escape):
        value = value[index]
    return key, value


def _check_adj_matrix(adj_matrix, copy=False):
    if sp.isspmatrix(adj_matrix):
        adj_matrix = adj_matrix.tocsr(
            copy=False).astype(np.float32, copy=copy)
    else:
        raise ValueError(f"Adjacency matrix must be in sparse format (got {type(adj_matrix)} instead).")

    assert adj_matrix.ndim == 2 and adj_matrix.shape[0] == adj_matrix.shape[1], adj_matrix.shape
    return adj_matrix


def _check_attr_matrix(attr_matrix, copy=False):
    if sp.isspmatrix(attr_matrix):
        attr_matrix = attr_matrix.astype(np.float32, copy=False).toarray()
    elif isinstance(attr_matrix, np.ndarray):
        attr_matrix = attr_matrix.astype(np.float32, copy=copy)
    else:
        raise ValueError(
            f"Attribute matrix must be a scipy.sparse.spmatrix or a np.ndarray (got {type(attr_matrix)} instead).")

    assert attr_matrix.ndim == 2, attr_matrix.shape
    return attr_matrix


def _check_label_matrix(label_matrix, copy=False):
    if sp.isspmatrix(label_matrix):
        label_matrix = label_matrix.toarray().astype(np.int32, copy=False).squeeze()
    else:
        label_matrix = np.array(label_matrix, dtype=np.int32, copy=copy).squeeze()

    label_matrix.ndim <= 2
    # For one-hot like matrix, convert it to 1D array
    if label_matrix.ndim == 2 and np.all(label_matrix.sum(1) == 1):
        label_matrix = label_matrix.argmax(1).astype(np.int32, copy=False)
    return label_matrix


def _check_edge_index(edge_index, copy=False):
    edge_index = np.array(edge_index, dtype=np.int64, copy=copy)
    return gf.asedge(edge_index, shape="col_wise")


def _check_edge_weight(edge_weight, copy=False):
    edge_weight = np.array(edge_index, dtype=np.float32, copy=copy)
    assert edge_weight.ndim == 1, edge_weight.ndim
    return edge_weight


def _check_dict(obj, copy=None):
    if not isinstance(obj, dict):
        raise ValueError("this should be a dict instance.")
    return obj


def _check_list_graphs(graphs, copy=False):
    # to avoid circular imports
    from .base_graph import BaseGraph
    for g in graphs:
        if not isinstance(g, BaseGraph):
            raise ValueError(f"graphs should be instances of 'BaseGraph', got ({type(g)} instead).")
    return tuple(g.copy() if copy else g for g in graphs)


def _check_dict_graphs(graphs, copy=False):
    # to avoid circular imports
    from .base_graph import BaseGraph
    for g in graphs.values():
        if not isinstance(g, BaseGraph):
            raise ValueError(f"graphs should be instances of 'BaseGraph', got ({type(g)} instead).")
    return gf.BunchDict({k: g.copy() if copy else g for k, g in graphs.items()})


_KEYS = ('adj_matrix', 'node_attr', 'node_label', 'node_graph_label',
         'edge_attr', 'edge_index', 'edge_weight', 'edge_label', 'edge_graph_label',
         'graph_attr', 'graph_label',
         'list_graphs', 'dict_graphs',
         'mapping', 'metadata')

# adj_matrix should be CSR matrix
# attribute matrices: node_attr, edge_attr, graph_attr should be 2D numpy array
# label matrices: node_label, node_graph_label, edge_label, graph_label should be 1D or 2D numpy array
# edge_index should be (2, N) numpy array
# edge_weight should be (N,) numpy array
_check_fn_dict = {'adj_matrix': _check_adj_matrix,
                  'node_attr': _check_attr_matrix,
                  'edge_attr': _check_attr_matrix,
                  'graph_attr': _check_attr_matrix,
                  'node_label': _check_label_matrix,
                  'node_graph_label': _check_label_matrix,
                  'edge_label': _check_label_matrix,
                  'edge_graph_label': _check_label_matrix,
                  'graph_label': _check_label_matrix,
                  'edge_index': _check_edge_index,
                  'edge_weight': _check_edge_weight,
                  "list_graphs": _check_list_graphs,
                  'dict_graphs': _check_dict_graphs,
                  'mapping': _check_dict,
                  'metadata': _check_dict}


def check_and_convert(key, value, multiple=False, copy=False):
    if value is not None:
        check_fn = _check_fn_dict.get(key, None)
        if not check_fn:
            warnings.warn(f"Unrecognized key {key}.")
            return key, value

        if multiple:
            if is_multiobjects(value):
                value = np.asarray([check_fn(v, copy=copy) for v in value], dtype=object)
            else:
                value = check_fn(value, copy=copy)
                if key != "graph_label":
                    # one graph, one label
                    value = np.asarray([value])
        else:
            value = check_fn(value, copy=copy)

    return key, value
