import warnings
import numpy as np
import graphgallery as gg
import scipy.sparse as sp
from typing import Optional
from .shape import maybe_shape, maybe_num_nodes

from ..functions import get_length

__all__ = ['remove_self_loops_edge',
           'add_selfloops_edge', 'segregate_self_loops_edge',
           'contains_self_loops_edge', 'add_remaining_self_loops',
           'normalize_edge', 'augment_edge',
           'asedge', 'edge_to_sparse_adj']


def normalize_edge(edge_index, edge_weight=None, rate=-0.5, fill_weight=1.0):
    edge_index = asedge(edge_index, shape="col_wise")

    num_nodes = edge_index.max() + 1

    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[1], dtype=gg.floatx())

    if fill_weight:
        edge_index, edge_weight = add_selfloops_edge(
            edge_index, edge_weight, num_nodes=num_nodes, fill_weight=fill_weight)

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
    """Augment a set of edges by connecting nodes to
        element in ``nbrs_to_link``.


    Parameters
    ----------
    edge_index: shape [M, 2] or [2, M]
        edge indices of a Scipy sparse adjacency matrix.
    nodes: the nodes that will be linked to the graph.
        list or np.array: the nodes connected to `nbrs_to_link`
        int: new added nodes connected to ``nbrs_to_link``, 
            node ids [num_nodes, ..., num_nodes+nodes-1].            
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
    -----
    Both ``nbrs_to_link`` and ``common_nbrs`` should NOT be specified together.

    See Also
    --------
    graphgallery.functional.augment_adj

    """

    if nbrs_to_link is not None and common_nbrs is not None:
        raise RuntimeError("Only one of them should be specified.")

    edge_index = asedge(edge_index, shape="col_wise")

    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[1], dtype=gg.floatx())

    num_nodes = edge_index.max() + 1

    if gg.is_intscalar(nodes):
        # int, add nodes to the graph
        nodes = np.arange(num_nodes, num_nodes + nodes, dtype=edge_index.dtype)
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
        num_repeat = len(common_nbrs)
        edges_to_link = np.hstack([np.vstack([np.tile(node, num_repeat), common_nbrs])
                                   for node in nodes])

    edges_to_link = np.hstack([edges_to_link, edges_to_link[[1, 0]]])
    added_edge_weight = np.zeros(edges_to_link.shape[1], dtype=edge_weight.dtype) + fill_weight

    augmented_edge_index = np.hstack([edge_index, edges_to_link])
    augmented_edge_weight = np.hstack([edge_weight, added_edge_weight])

    return augmented_edge_index, augmented_edge_weight


def contains_self_loops_edge(edge_index):
    r"""Returns `True` if the graph given by `edge_index` contains self-loops.
    """
    edge_index = asedge(edge_index, shape="col_wise")
    mask = edge_index[0] == edge_index[1]
    return mask.sum().item() > 0


def remove_self_loops_edge(edge_index: np.ndarray, edge_weight: Optional[np.ndarray] = None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    """
    edge_index = asedge(edge_index, shape="col_wise")
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_weight is None:
        return edge_index, np.ones(edge_index.shape[1], dtype=gg.floatx())
    else:
        return edge_index, edge_weight[mask]


def segregate_self_loops_edge(edge_index, edge_weight: Optional[np.ndarray] = None):
    r"""Segregates self-loops from the graph.
    """

    edge_index = asedge(edge_index, shape="col_wise")
    mask = edge_index[0] != edge_index[1]
    inv_mask = ~mask

    loop_edge_index = edge_index[:, inv_mask]
    loop_edge_weight = None if edge_weight is None else edge_weight[inv_mask]
    edge_index = edge_index[:, mask]
    edge_weight = None if edge_weight is None else edge_weight[mask]

    return edge_index, edge_weight, loop_edge_index, loop_edge_weight


def add_selfloops_edge(edge_index: np.ndarray, edge_weight: Optional[np.ndarray] = None,
                       num_nodes: Optional[int] = None, fill_weight: float = 1.0):
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted, self-loops will be added with edge weights
    denoted by :obj:`fill_value`.
    """
    edge_index = asedge(edge_index, shape="col_wise")
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[1], dtype=gg.floatx())

    diagnal_edge_index = np.asarray(np.diag_indices(num_nodes)).astype(edge_index.dtype, copy=False)

    updated_edge_index = np.hstack([edge_index, diagnal_edge_index])

    diagnal_edge_weight = np.zeros(num_nodes, dtype=gg.floatx()) + fill_weight
    updated_edge_weight = np.hstack([edge_weight, diagnal_edge_weight])

    return updated_edge_index, updated_edge_weight


def add_remaining_self_loops(edge_index: np.ndarray,
                             edge_weight: Optional[np.ndarray] = None,
                             fill_value: float = 1.,
                             num_nodes: Optional[int] = None):
    r"""Adds remaining self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted and already contains a few self-loops, only
    non-existent self-loops will be added with edge weights denoted by
    :obj:`fill_value`.

    """
    edge_index = asedge(edge_index, shape="col_wise")
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    mask = row != col

    loop_index = np.asarray(np.diag_indices(num_nodes)).astype(edge_index.dtype, copy=False)
    edge_index = np.hstack([edge_index[:, mask], loop_index])

    if edge_weight is not None:
        inv_mask = ~mask
        loop_weight = np.full((num_nodes, ), fill_value, dtype=edge_weight.dtype)
        remaining_edge_weight = edge_weight[inv_mask]
        if remaining_edge_weight.size > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight
        edge_weight = np.hstack([edge_weight[mask], loop_weight])

    return edge_index, edge_weight


def asedge(edge: np.ndarray, shape="col_wise", symmetric=False, dtype=None):
    """make sure the array as edge like,
    shape [M, 2] or [2, M] with dtype 'dtype' (or 'int64')
    if ``symmetric=True``, it wiil have shape
    [2*M, 2] or [2, M*2].

    Parameters
    ----------
    edge : List, np.ndarray
        edge like list or array
    shape : str, optional
        row_wise: edge has shape [M, 2]
        col_wise: edge has shape [2, M]
        by default ``col_wise``
    symmetric: bool, optional
        if ``True``, the output edge will be 
        symmectric, i.e., 
        row_wise: edge has shape [2*M, 2]
        col_wise: edge has shape [2, M*2]
        by default ``False``
    dtype: string, optional
        data type for edges, if None, default to 'int64'

    Returns
    -------
    np.ndarray
        edge array
    """
    assert shape in ["row_wise", "col_wise"], shape
    assert isinstance(edge, (np.ndarray, list, tuple)), edge
    edge = np.asarray(edge, dtype=dtype or "int64")
    assert edge.ndim == 2 and 2 in edge.shape, edge.shape
    N, M = edge.shape
    if N == M == 2 and shape == "col_wise":
        # TODO: N=M=2 is confusing, we assume that edge was 'row_wise'
        warnings.warn(f"The shape of the edge is {N}x{M}."
                      f"we assume that {edge} was 'row_wise'")
        edge = edge.T
    elif (shape == "col_wise" and N != 2) or (shape == "row_wise" and M != 2):
        edge = edge.T

    if symmetric:
        if shape == "col_wise":
            edge = np.hstack([edge, edge[[1, 0]]])
        else:
            edge = np.vstack([edge, edge[:, [1, 0]]])

    return edge


def edge_to_sparse_adj(edge: np.ndarray,
                       edge_weight: Optional[np.ndarray] = None,
                       shape: Optional[tuple] = None) -> sp.csr_matrix:
    """Convert (edge, edge_weight) representation to a Scipy sparse matrix

    Parameters
    ----------
    edge : list or np.ndarray
        edge index of sparse matrix, shape [2, M]
    edge_weight : Optional[np.ndarray], optional
        edge weight of sparse matrix, shape [M,], by default None
    shape : Optional[tuple], optional
        shape of sparse matrix, by default None

    Returns
    -------
    scipy.sparse.csr_matrix

    """

    edge = asedge(edge, shape="col_wise")

    if edge_weight is None:
        edge_weight = np.ones(edge.shape[1], dtype=gg.floatx())

    if shape is None:
        shape = maybe_shape(edge)
    return sp.csr_matrix((edge_weight, edge), shape=shape)
