import numpy as np
import scipy.sparse as sp

from typing import Union
from graphgallery.utils.shape import get_length
from graphgallery.transformers import edge_transpose
from graphgallery import is_interger_scalar

def augment_edge(edge_index: np.ndarray, nodes: Union[list, int, np.ndarray], 
                 edge_weight: np.ndarray=None, *,
                 nbrs_to_link: Union[list, np.ndarray, None]=None, 
                 common_nbrs: Union[list, np.ndarray, None]=None,
                 fill_weight: float=1.0) -> tuple:
    """Augment a set of edges by linking nodes to
        each element in `nbrs_to_link`.
    
                                
    Parameters
    ----------
    edge_index: shape [M, 2] or [2, M] -> [2, M]
            edge indices of a Scipy sparse adjacency matrix.
    nodes: the nodes that will be linked to the graph.
        list or np.array: the nodes connected to `nbrs_to_link`
        int: new added nodes connected to `nbrs_to_link`, 
            node ids [n_nodes, ..., n_nodes+nodes-1].            
    edge_weight: shape [M,]
        edge weights of a Scipy sparse adjacency matrix.
    nbrs_to_link: a list of N elements,
        where N is the length of 'nodes'.
        the specified neighbor(s) for each added node.
        if `None`, it will be set to `[0, ..., N-1]`.
    common_nbrs: shape [None,].
        specified common neighbors for each added node.
    fill_weight: edge weight for the augmented edges.
    
    NOTE:
    ----------
    Both `nbrs_to_link` and `common_nbrs` should not be specified together.
    
    See Also
    ----------
    graphgallery.transformers.augment_adj
        
    """
    
    if nbrs_to_link is not None and common_nbrs is not None:
        raise RuntimeError("Only one of them should be specified.")
        
    edge_index = edge_transpose(edge_index)
    
    if edge_weight is None:
        edge_weight = np.ones(edge_index.shape[1], dtype=floatx())    
        
    n_nodes = edge_index.max() + 1
    
    if is_interger_scalar(nodes):
        # int, add nodes to the graph
        nodes = np.arange(n_nodes, n_nodes + nodes, dtype=edge_index.dtype)
    else:
        # array-like, link nodes to the graph
        nodes = np.asarray(nodes, dtype=edge_index.dtype)

    if common_nbrs is None and nbrs_to_link is None:
        nbrs_to_link = np.arange(nodes.size, dtype=edge_index.dtype)
    
    if not nodes.size == len(nbrs_to_link):
        raise ValueError("The length of 'nbrs_to_link' should equal to 'nodes'.")

    if nbrs_to_link is not None:
        edges_to_link = np.hstack([np.vstack([np.tile(node, get_length(nbr)), nbr]) 
                                            for node, nbr in zip(nodes, nbrs_to_link)]) 
    else:
        n_repeat = len(common_nbrs)
        edges_to_link = np.hstack([np.vstack([np.tile(node, n_repeat), common_nbrs]) 
                                 for node in nodes])
        
    edges_to_link = np.hstack([edges_to_link, edges_to_link[[1,0]]])
    added_edge_weight = np.zeros(edges_to_link.shape[1], dtype=edge_weight.dtype) + fill_weight

    augmented_edge_index = np.hstack([edge_index, edges_to_link])
    augmented_edge_weight = np.hstack([edge_weight, added_edge_weight])
    
    return augmented_edge_index, augmented_edge_weight


