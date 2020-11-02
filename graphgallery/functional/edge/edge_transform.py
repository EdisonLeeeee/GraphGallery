import numpy as np
import scipy.sparse as sp
import graphgallery as gg

from typing import Union, Optional

from .to_adj import edge_transpose
from ..ops import get_length


__all__ = ['add_selfloops_edge', 'normalize_edge', 'augment_edge']


def add_selfloops_edge(edge_index, edge_weight, n_nodes=None, fill_weight=1.0):
    edge_index = edge_transpose(edge_index)

    if n_nodes is None:
        n_nodes = edge_index.max() + 1

    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[1], dtype=gg.floatx())

    diagnal_edge_index = np.asarray(np.diag_indices(n_nodes)).astype(edge_index.dtype, copy=False)

    updated_edge_index = np.hstack([edge_index, diagnal_edge_index])

    diagnal_edge_weight = np.zeros(n_nodes, dtype=gg.floatx()) + fill_weight
    updated_edge_weight = np.hstack([edge_weight, diagnal_edge_weight])

    return updated_edge_index, updated_edge_weight


def normalize_edge(edge_index, edge_weight=None, rate=-0.5, fill_weight=1.0):
    edge_index = edge_transpose(edge_index)

    n_nodes = edge_index.max() + 1

    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[1], dtype=gg.floatx())

    if fill_weight:
        edge_index, edge_weight = add_selfloops_edge(
            edge_index, edge_weight, n_nodes=n_nodes, fill_weight=fill_weight)

    degree = np.bincount(edge_index[0], weights=edge_weight)
    degree_power = np.power(degree, rate, dtype=gg.floatx())
    row, col = edge_index
    edge_weight_norm = degree_power[row] * edge_weight * degree_power[col]

    return edge_index, edge_weight_norm


def augment_edge(edge_index: np.ndarray, nodes: np.ndarray,
                 edge_weight: np.ndarray = None, *,
                 nbrs_to_link: Optional[np.ndarray] = None,
                 common_nbrs: Optional[np.ndarray] = None,
                 fill_weight: float = 1.0) -> tuple:
    """Augment a set of edges by linking nodes to
        each element in `nbrs_to_link`.


    Parameters
    ----------
    edge_index: shape [M, 2] or [2, M] -> [2, M]
            edge indices of a Scipy sparse adjacency matrix.
    nodes: the nodes that will be linked to the graph.
        list or np.array: the nodes connected to `nbrs_to_link`
        int: new added nodes connected to `nbrs_to_link`, 
            node ids [n_nodes, ..., n_nodes+nodes-1].            
    edge_weight: shape [M,]
        edge weights of a Scipy sparse adjacency matrix.
    nbrs_to_link: a list of N elements,
        where N is the length of 'nodes'.
        the specified neighbor(s) for each added node.
        if `None`, it will be set to `[0, ..., N-1]`.
    common_nbrs: shape [None,].
        specified common neighbors for each added node.
    fill_weight: edge weight for the augmented edges.

    NOTE:
    ----------
    Both `nbrs_to_link` and `common_nbrs` should not be specified together.

    See Also
    ----------
    graphgallery.functional.augment_adj

    """

    if nbrs_to_link is not None and common_nbrs is not None:
        raise RuntimeError("Only one of them should be specified.")

    edge_index = edge_transpose(edge_index)

    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[1], dtype=gg.floatx())

    n_nodes = edge_index.max() + 1

    if gg.is_intscalar(nodes):
        # int, add nodes to the graph
        nodes = np.arange(n_nodes, n_nodes + nodes, dtype=edge_index.dtype)
    else:
        # array-like, link nodes to the graph
        nodes = np.asarray(nodes, dtype=edge_index.dtype)

    if common_nbrs is None and nbrs_to_link is None:
        nbrs_to_link = np.arange(nodes.size, dtype=edge_index.dtype)

    if not nodes.size == len(nbrs_to_link):
        raise ValueError("The length of 'nbrs_to_link' should equal to 'nodes'.")

    if nbrs_to_link is not None:
        edges_to_link = np.hstack([np.vstack([np.tile(node, get_length(nbr)), nbr])
                                   for node, nbr in zip(nodes, nbrs_to_link)])
    else:
        n_repeat = len(common_nbrs)
        edges_to_link = np.hstack([np.vstack([np.tile(node, n_repeat), common_nbrs])
                                   for node in nodes])

    edges_to_link = np.hstack([edges_to_link, edges_to_link[[1, 0]]])
    added_edge_weight = np.zeros(edges_to_link.shape[1], dtype=edge_weight.dtype) + fill_weight

    augmented_edge_index = np.hstack([edge_index, edges_to_link])
    augmented_edge_weight = np.hstack([edge_weight, added_edge_weight])

    return augmented_edge_index, augmented_edge_weight
