import networkx as nx
from .property import is_directed

def from_nxgraph(G):
    return nx.to_scipy_sparse_matrix(G).astype('float32')

def to_nxgraph(G, directed=None):
    if directed is None:
        directed = is_directed(G)
    if directed:
        create_using = nx.DiGraph
    else:
        create_using = nx.Graph
    return nx.from_scipy_sparse_matrix(G, create_using=create_using)