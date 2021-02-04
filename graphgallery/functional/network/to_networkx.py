import networkx as nx
from .property import is_directed

__all__ = ["from_nxgraph", "to_nxgraph"]


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
