import numpy as np
import scipy.sparse as sp
from typing import Union, Optional, List, Tuple, Any
from ..data_type import is_objects

__all__ = ['sparse_collate', 'check_and_convert', 'index_select']

_SPARSE_THRESHOLD = 0.5


def sparse_collate(key, val):
    # TODO: multiple graph
    if is_objects(val):
        return key, val

    if isinstance(val, np.ndarray) and val.ndim == 2:
        # one-hot like matrix stored with 1D array
        if "labels" in key and np.all(val.sum(1) == 1):
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
            if is_objects(value):
                value = np.asarray([check_fn(v, copy=copy) for v in value])
            else:
                value = check_fn(value, copy=copy)
                if not "graph" in key:
                    value = np.asarray([value])
        else:
            value = check_fn(value, copy=copy)

    return key, value
