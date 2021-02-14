"""Modified from networkx.
Generators for some classic graphs.
"""
# TODO: implement with sicpy sparse matrix
import scipy.sparse as sp
import networkx as nx
from .transform import from_nxgraph

__all__ = ['complete_graph',
           'newman_watts_strogatz_graph',
           ]


def complete_graph(n, directed=False):
    """Return the complete graph `K_n` with n nodes.

    A complete graph on `n` nodes means that all pairs
    of distinct nodes have an edge connecting them.

    Parameters
    ----------
    n : int or iterable container of nodes
        If n is an integer, nodes are from range(n).
        If n is a container of nodes, those nodes appear in the graph.
    directed : where to construct a directed graph

    Examples
    --------
    >>> G = gf.complete_graph(9)
    >>> G = gf.complete_graph(range(11, 14))
    >>> G = gf.complete_graph(4, directed=True)
    """
    if directed:
        create_using = nx.DiGraph()
    else:
        create_using = nx.Graph()

    G = nx.complete_graph(n, create_using=create_using)
    return from_nxgraph(G)


def newman_watts_strogatz_graph(n, k, p, seed=None):
    """Returns a Newman–Watts–Strogatz small-world graph.

    Parameters
    ----------
    n : int
        The number of nodes.
    k : int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of adding a new edge for each edge.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.

    Notes
    -----
    First create a ring over $n$ nodes [1]_.  Then each node in the ring is
    connected with its $k$ nearest neighbors (or $k - 1$ neighbors if $k$
    is odd).  Then shortcuts are created by adding new edges as follows: for
    each edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest
    neighbors" with probability $p$ add a new edge $(u, w)$ with
    randomly-chosen existing node $w$.  In contrast with
    :func:`watts_strogatz_graph`, no edges are removed.

    Example
    -------
    >>> gf.newman_watts_strogatz_graph(100, 20, 0.05)
    <100x100 sparse matrix of type '<class 'numpy.float32'>'
        with 2078 stored elements in Compressed Sparse Row format>    

    See Also
    --------
    nx.newman_watts_strogatz_graph

    References
    ----------
    .. [1] M. E. J. Newman and D. J. Watts,
       Renormalization group analysis of the small-world network model,
       Physics Letters A, 263, 341, 1999.
       https://doi.org/10.1016/S0375-9601(99)00757-4
    """
    G = nx.newman_watts_strogatz_graph(n, k, p, seed=seed)
    return from_nxgraph(G)
