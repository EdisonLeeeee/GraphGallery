import numpy as np
import scipy.sparse as sp

from typing import Union
from graphgallery.utils.shape import get_length
from graphgallery.transformers import augment_edge


def augment_adj(adj_matrix: sp.csr_matrix, N: int, *,
                nbrs_per_node: Union[list, np.ndarray, None]=None, 
                common_nbrs: Union[list, np.ndarray, None]=None,
                fill_weight: float=1.0) -> sp.csr_matrix:
    """Augment a specified adjacency matrix by adding N nodes.
    
    Examples
    ----------
    # add 2 nodes, which are adjacent to [2,3] and 3, respectively.
    >>> augmented_adj = augment_adj(adj_matrix, N=2, 
                                nbrs_per_node=[[2,3],3], 
                                fill_weight=1.0)
                                
                                
    # add 2 nodes, all adjacent to [1,2,3].
    >>> augmented_adj = augment_adj(adj_matrix, N=2, 
                                common_nbrs=[1,2,3], 
                                fill_weight=1.0)  
                                
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
    fill_weight: edge weight for the added edges.
    
    NOTE:
    ----------
    Both `nbrs_per_node` and `common_nbrs` should not be specified together.
    
    
    See Also
    ----------
    graphgallery.transformers.augment_edge    
        
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
    edge_index = adj_matrix.row, adj_matrix.col

    augmented_edge_index, augmented_edge_weight = augment_edge(edge_index, N,
                                                               edge_weight=adj_matrix.data,
                                                               nbrs_per_node=nbrs_per_node,
                                                               common_nbrs=common_nbrs,
                                                               fill_weight=fill_weight)
    
    
    augmented_adj = sp.csr_matrix((augmented_edge_weight, augmented_edge_index), 
                               shape=(n_augmented_nodes, n_augmented_nodes))
    
    augmented_adj.eliminate_zeros()
    return augmented_adj


