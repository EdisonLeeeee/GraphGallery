import numpy as np
import scipy.sparse as sp

from typing import Union
from graphgallery.utils.shape import get_length
from graphgallery.transformers import edge_transpose

def augment_edge(edge_index: np.ndarray, N: int, edge_weight: np.ndarray=None, *,
                nbrs_per_node: Union[list, np.ndarray, None]=None, 
                common_nbrs: Union[list, np.ndarray, None]=None,
                fill_weight: float=1.0) -> sp.csr_matrix:
    """Augment a set of edges by adding N nodes.
    
                                
    Parameters
    ----------
    edge_index: shape [M, 2] or [2, M] -> [2, M]
            edge indices of a Scipy sparse adjacency matrix.
    N: number of added nodes.
        node ids [n_nodes, ..., n_nodes+N-1].            
    edge_weight: shape [M,]
            edge weights of a Scipy sparse adjacency matrix.
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
    graphgallery.transformers.augment_adj
        
    """
    
    if nbrs_per_node is not None and common_nbrs is not None:
        raise RuntimeError("Only one of them should be specified.")
        
    if common_nbrs is None:
        if nbrs_per_node is None:
            nbrs_per_node = range(N)
        elif len(nbrs_per_node) != N:
            raise ValueError("The neighbors of each node should be specified!")
            
    edge_index = edge_transpose(edge_index)
    
    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[1], dtype=floatx())    

    n_nodes = edge_index.max() + 1
    n_augmented_nodes = n_nodes + N
    added_nodes = range(n_nodes, n_augmented_nodes)

    if nbrs_per_node is not None:
        added_edges = np.vstack([added_nodes, nbrs_per_node])
        added_edges = np.hstack([np.vstack([np.tile(node, get_length(nbr)), nbr]) 
                                            for node, nbr in zip(added_nodes, nbrs_per_node)]) 
    else:
        n_repeat = len(common_nbrs)
        added_edges = np.hstack([np.vstack([np.tile(node, n_repeat), common_nbrs]) 
                                 for node in added_nodes])
        
    added_edges_T = added_edges[[1,0]]
    added_edge_weight = np.zeros(added_edges.shape[1]*2, dtype=edge_weight.dtype) + fill_weight

    augmented_edge_index = np.hstack([edge_index, added_edges, added_edges_T])
    augmented_edge_weight = np.hstack([edge_weight, added_edge_weight])
    
    return augmented_edge_index, augmented_edge_weight


