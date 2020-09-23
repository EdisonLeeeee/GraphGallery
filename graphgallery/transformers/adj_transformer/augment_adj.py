import numpy as np
import scipy.sparse as sp

from typing import Union
from graphgallery.utils.shape import get_length

def augment_adj(adj_matrix: sp.csr_matrix, N: int, *,
                nbrs_per_node: Union[list, np.ndarray, None]=None, 
                common_nbrs: Union[list, np.ndarray, None]=None,
                weight: float=1.0) -> sp.csr_matrix:
    """Augment a specified adjacency matrix by adding N nodes.
    
    Examples
    ----------
    # add 2 nodes, which are adjacent to [2,3] and 3, respectively.
    >>> augmented_adj = augment_adj(adj_matrix, N=2, 
                                nbrs_per_node=[[2,3],3], 
                                weight=1.0)
                                
                                
    # add 2 nodes, all adjacent to [1,2,3].
    >>> augmented_adj = augment_adj(adj_matrix, N=2, 
                                common_nbrs=[1,2,3], 
                                weight=1.0)  
                                
    Parameters
    ----------
    adj_matrix: shape [n_nodes, n_nodes].
        A Scipy sparse adjacency matrix.
    N: number of added nodes.
        node ids [n_nodes, ..., n_nodes+N-1].
    nbrs_per_node: shape [N,].
        the specified neighbor(s) for each added node.
        if `None`, it will be set to `[0, ..., N-1]`.
    common_nbrs: shape [None,].
        specified common neighbors for each added node.
    weight: edge weight for the added edges.
    
    NOTE:
    ----------
    Both `nbrs_per_node` and `common_nbrs` should not be specified together.
        
    """
    
    if nbrs_per_node is not None and common_nbrs is not None:
        raise RuntimeError("Only one of them should be specified.")
        
    if common_nbrs is None:
        if nbrs_per_node is None:
            nbrs_per_node = range(N)
        elif len(nbrs_per_node) != N:
            raise ValueError("The neighbors of each node should be specified!")
        
    n_nodes = adj_matrix.shape[0]
    n_augmented_nodes = n_nodes + N
    added_nodes = range(n_nodes, n_augmented_nodes)

    adj_matrix = adj_matrix.tocoo(copy=False)
    edges = adj_matrix.row, adj_matrix.col

    if nbrs_per_node is not None:
        added_edges = np.vstack([added_nodes, nbrs_per_node])
        added_edges = np.hstack([np.vstack([np.tile(node, get_length(nbr)), nbr]) 
                                            for node, nbr in zip(added_nodes, nbrs_per_node)]) 
    else:
        n_repeat = len(common_nbrs)
        added_edges = np.hstack([np.vstack([np.tile(node, n_repeat), common_nbrs]) 
                                 for node in added_nodes])
        
    added_edges_T = added_edges[[1,0]]
    added_edge_weight = np.zeros(added_edges.shape[1]*2, dtype=adj_matrix.dtype) + weight

    augmented_edges = np.hstack([edges, added_edges, added_edges_T])
    augmented_data = np.hstack([adj_matrix.data, added_edge_weight])
    
    augmented_adj = sp.csr_matrix((augmented_data, augmented_edges), 
                               shape=(n_augmented_nodes, n_augmented_nodes))
    
    augmented_adj.eliminate_zeros()
    return augmented_adj


