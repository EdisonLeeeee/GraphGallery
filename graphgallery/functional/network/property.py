import numpy as np
import scipy.sparse as sp
from graphgallery.data_type import is_multiobjects
from ..decorators import multiple


@multiple()
def degree(A):
    assert A is not None

    if not is_directed(A):
        return A.sum(1).A1
    else:
        # in-degree and out-degree
        return A.sum(0).A1, A.sum(1).A1


@multiple()
def largest_connected_components(A):

    assert A is not None

    _, component_indices = sp.csgraph.connected_components(A)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[-1]
    nodes_to_keep = np.where(component_indices == components_to_keep)[0]
    return nodes_to_keep


def is_directed(A) -> bool:
    assert A is not None
    if is_multiobjects(A):
        return any(is_directed(adj) for adj in A)
    return (A != A.T).sum() != 0


def has_singleton(A) -> bool:
    assert A is not None
    if is_multiobjects(A):
        return any(has_singleton(adj) for adj in A)
    out_deg = A.sum(1).A1
    in_deg = A.sum(0).A1
    return np.any(np.logical_and(in_deg == 0, out_deg == 0))


def has_selfloops(A) -> bool:
    assert A is not None
    if is_multiobjects(A):
        return any(has_selfloops(adj) for adj in A)
    return not np.allclose(A.diagonal(), 0)


def is_binary(A) -> bool:
    assert A is not None
    if is_multiobjects(A):
        return all(is_binary(adj) for adj in A)
    return np.all(np.unique(A) == (0, 1))


def is_symmetric(A) -> bool:
    assert A is not None
    if is_multiobjects(A):
        return all(is_symmetric(adj) for adj in A)
    return np.abs(A - A.T).sum() == 0


def is_weighted(A) -> bool:
    assert A is not None
    if is_multiobjects(A):
        return any(is_weighted(adj) for adj in A)
    return np.any(A.data != 1)


def is_connected(A) -> bool:
    """Returns True if the graph is connected, False otherwise.
    For directed graph, it test the weak connectivity.

    A directed graph is weakly connected if and only if the graph
    is connected when the direction of the edge between nodes is ignored.

    Note that if a graph is strongly connected (i.e. the graph is connected
    even when we account for directionality), it is by definition weakly
    connected as well.

    Example
    -------
    >>> G = np.array([[0,1,1], [0,0,0], [0,0,0]])
    >>> G
    array([[0, 1, 1],
       [0, 0, 0],
       [0, 0, 0]])
    >>> G = sp.csr_matrix(G)
    >>> gf.is_connected(G)
    True
    """
    assert A is not None
    if is_multiobjects(A):
        return all(is_connected(adj) for adj in A)
    return sp.csgraph.connected_components(A, directed=is_directed(A), return_labels=False, connection='weak') == 1


def is_strong_connected(A) -> bool:
    """Test directed graph for strong connectivity.

    A directed graph is strongly connected if and only if every vertex in
    the graph is reachable from every other vertex.

    Example
    -------
    >>> G = np.array([[0,1,1], [0,0,0], [0,0,0]])
    >>> G
    array([[0, 1, 1],
       [0, 0, 0],
       [0, 0, 0]])
    >>> G = sp.csr_matrix(G)
    >>> gf.is_strong_connected(G)
    False
    """
    assert A is not None
    if is_multiobjects(A):
        return all(is_strong_connected(adj) for adj in A)
    return sp.csgraph.connected_components(A, directed=is_directed(A), return_labels=False, connection='strong') == 1


def is_eulerian(A):
    """Returns True if and only if `A` is Eulerian.

    A graph is *Eulerian* if it has an Eulerian circuit. An *Eulerian
    circuit* is a closed walk that includes each edge of a graph exactly
    once.

    Parameters
    ----------
    A : Scipy sparse matrix
       A graph, either directed or undirected.

    Examples
    --------
    >>> gf.is_eulerian(nx.to_scipy_sparse_matrix(nx.DiGraph({0: [3], 1: [2], 2: [3], 3: [0, 1]})))
    True
    >>> gf.is_eulerian(nx.to_scipy_sparse_matrix(nx.complete_graph(5)))
    True
    >>> gf.is_eulerian(nx.to_scipy_sparse_matrix(nx.petersen_graph()))
    False


    Notes
    -----
    If the graph is not connected (or not strongly connected, for
    directed graphs), this function returns False.

    See Also
    --------
    networkx.is_eulerian
    """
    assert A is not None
    if is_multiobjects(A):
        return all(is_eulerian(adj) for adj in A)

    if not is_connected(A):
        return False

    deg = degree(A)
    if isinstance(deg, tuple):
        # Directed graph
        # Every node must have equal in degree and out degree and the
        # graph must be strongly connected
        ind, outd = deg
        return np.all(ind == outd)
    # An undirected Eulerian graph has no vertices of odd degree and
    # must be connected.
    return np.all(deg % 2 == 0)
