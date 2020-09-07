import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from functools import lru_cache
from copy import copy as copy_fn

from graphgallery.data.preprocess import largest_connected_components
from graphgallery.data.base_graph import base_graph


def _check_and_convert(adj_matrix, attr_matrix, labels, copy=True):
    # Make sure that the dimensions of matrices / arrays all agree
    if adj_matrix is not None:
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr(
                copy=False).astype(np.float32, copy=copy)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)"
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree!")

    if attr_matrix is not None:
        if sp.isspmatrix(attr_matrix):
            attr_matrix = attr_matrix.tocsr(
                copy=False).astype(np.float32, copy=copy)
        elif isinstance(attr_matrix, np.ndarray):
            attr_matrix = attr_matrix.astype(np.float32, copy=copy)
        else:
            raise ValueError(
                "Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)".format(type(attr_matrix)))

        if attr_matrix.shape[0] != adj_matrix.shape[0]:
            raise ValueError(
                "Dimensions of the adjacency and attribute matrices don't agree!")

    if labels is not None:
        labels = np.array(labels, dtype=np.int32, copy=copy)
        if labels.ndim != 1:
            labels = labels.argmax(1)
        # if labels.shape[0] != adj_matrix.shape[0]:
            # raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree!")

    return adj_matrix, attr_matrix, labels


class MuitiGraph(base_graph):
    """Attributed labeled multigraph stored in sparse matrix form."""
    ...
