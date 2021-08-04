import numpy as np
import networkx as nx
import scipy.sparse as sp
from .property import is_directed, is_weighted
from ..decorators import multiple

__all__ = ["from_nxgraph", "to_nxgraph", "to_directed",
           "to_undirected", "to_unweighted"]


@multiple()
def from_nxgraph(G):
    """Convert a networkx graph to scipy sparse matrix (CSR)

    Parameters
    ----------
    G : networkx graph
        a networkx graph

    Returns
    -------
    scipy.sparse.csr_matrix
        Scipy sparse matrix with CSR format
    """
    return nx.to_scipy_sparse_matrix(G).astype('float32')


@multiple()
def to_nxgraph(G, directed=None):
    """Convert Scipy sparse matrix to networkx graph to

    Parameters
    ----------
    G : Scipy sparse matrix
        a Scipy sparse matrix
    directed : bool, optional
        whether convert to a directed graph, by default None,
        if checks if the graph is directed and convert it to propert type

    Returns
    -------
    networkx graph  
        a netwotkx graph
    """
    if directed is None:
        directed = is_directed(G)
    if directed:
        create_using = nx.DiGraph
    else:
        create_using = nx.Graph
    return nx.from_scipy_sparse_matrix(G, create_using=create_using)


@multiple()
def to_undirected(A):
    """Convert to an undirected graph (make adjacency matrix symmetric)."""
    if is_weighted(A):
        raise RuntimeError(
            "Convert to unweighted graph first."
        )
    A = A.maximum(A.T)
    return A

@multiple()
def to_directed(A):
    """Convert to a directed graph."""
    if is_directed(A):
        return A.copy()
    A = sp.triu(A)
    return A

@multiple()
def to_unweighted(A):
    """Convert to an unweighted graph (set all edge weights to 1)."""
    A = sp.csr_matrix(
        (np.ones_like(A.data), A.indices, A.indptr), shape=A.shape)
    return A
