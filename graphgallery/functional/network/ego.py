import numpy as np

from numba import njit
from numba import types
from numba.typed import Dict

from typing import Union, Tuple
from ..edge_level import asedge


__all__ = ['ego_graph']


@njit
def extra_edges(indices, indptr,
                last_level, seen,
                hops: int):
    edges = []
    mapping = Dict.empty(
        key_type=types.int64,
        value_type=types.int64,
    )
    for u in last_level:
        nbrs = indices[indptr[u]:indptr[u + 1]]
        nbrs = nbrs[seen[nbrs] == hops]
        mapping[u] = 1
        for v in nbrs:
            if not v in mapping:
                edges.append((u, v))
    return edges


def ego_graph(adj_matrix, targets, hops: int = 1):
    """Returns induced subgraph of neighbors centered at node n within
    a given radius.

    Parameters
    ----------
    adj_matrix : A Scipy sparse adjacency matrix
        representing a graph

    targets : Center nodes
        A single node or a list of nodes

    hops : number, optional
        Include all neighbors of distance<=hops from nodes.

    Returns
    -------
    (edges, nodes):
        edges: shape [2, M], the edges of the subgraph
        nodes: shape [N], the nodes of the subgraph

    Notes
    -----
    This is a faster implementation of 
    `networkx.ego_graph`


    See Also
    --------
    networkx.ego_graph

    """

    if np.ndim(targets) == 0:
        targets = [targets]
    elif isinstance(targets, np.ndarray):
        targets = targets.tolist()
    else:
        targets = list(targets)

    indices = adj_matrix.indices
    indptr = adj_matrix.indptr

    edges = {}
    start = 0
    N = adj_matrix.shape[0]
    seen = np.zeros(N) - 1
    seen[targets] = 0
    for level in range(hops):
        end = len(targets)
        while start < end:
            head = targets[start]
            nbrs = indices[indptr[head]:indptr[head + 1]]
            for u in nbrs:
                if seen[u] < 0:
                    targets.append(u)
                    seen[u] = level + 1
                if (u, head) not in edges:
                    edges[(head, u)] = level + 1

            start += 1

    if len(targets[start:]):
        e = extra_edges(indices, indptr, np.array(targets[start:]), seen, hops)
    else:
        e = []

    return asedge(list(edges.keys()) + e, shape='row_wise'), np.asarray(targets)
