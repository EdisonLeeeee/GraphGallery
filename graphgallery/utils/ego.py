import numpy as np

from numba import njit
from numba import types
from numba.typed import Dict

@njit
def extra_edges(indices, indptr, last_level, seen, hops):
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

def ego_graph(adj, targets, hops=1):
    """Returns induced subgraph of neighbors centered at node n within
    a given radius.

    Arguments
    ----------
    adj : A Scipy sparse adjacency matrix
        representing a graph

    targets : Center nodes
        A single node or a list of nodes

    hops : number, optional
        Include all neighbors of distance<=hops from nodes.

    Notes
    --------
    This is a faster implementation of 
    `networkx.ego_graph`
    
        
    See Also
    --------
    networkx.ego_graph
    
    """
    
    if np.ndim(targets) == 0:
        nodes = [targets]
        
    indices = adj.indices
    indptr = adj.indptr
    
    edges = {}
    start = 0
    N = adj.shape[0]
    seen = np.zeros(N)-1
    seen[nodes] = 0
    for level in range(hops):
        end = len(nodes)
        while start < end:
            head = nodes[start]
            nbrs = indices[indptr[head]:indptr[head + 1]]
            for u in nbrs:
                if seen[u] < 0:
                    nodes.append(u)
                    seen[u] = level + 1
                if (u, head) not in edges:
                    edges[(head, u)] = level + 1

            start += 1
            
    if len(nodes[start:]):
        e = extra_edges(indices, indptr, np.array(nodes[start:]), seen, hops)
    else:
        e = []
    
    return np.asarray(list(edges.keys())  + e), np.asarray(nodes)
