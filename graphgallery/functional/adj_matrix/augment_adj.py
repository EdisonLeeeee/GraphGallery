import numpy as np
import scipy.sparse as sp

from typing import Union
from ..edge_level import augment_edge
from ..ops import get_length


def augment_adj(adj_matrix: sp.csr_matrix, nodes: Union[list, int, np.ndarray],
                edge_weight: np.ndarray = None, *,
                nbrs_to_link: Union[list, np.ndarray, None] = None,
                common_nbrs: Union[list, np.ndarray, None] = None,
                fill_weight: float = 1.0) -> sp.csr_matrix:
    """Augment a specified adjacency matrix by linking nodes to
        each element in `nbrs_to_link`.

    Examples
    ----------
    # add 2 nodes adjacent to [2,3] and 3, respectively.
    >>> augmented_adj = augment_adj(adj_matrix, nodes=2, 
                                nbrs_to_link=[[2,3],3], 
                                fill_weight=1.0)


    # add 2 nodes all adjacent to [1,2,3].
    >>> augmented_adj = augment_adj(adj_matrix, nodes=2, 
                                common_nbrs=[1,2,3], 
                                fill_weight=1.0)  

     # add 3 edges, [3,1], [4,2], [5,3].
    >>> augmented_adj = augment_adj(adj_matrix, nodes=[3,4,5], 
                                common_nbrs=[1,2,3], 
                                fill_weight=1.0)                                 
    Parameters
    ----------
    adj_matrix: shape [num_nodes, num_nodes].
        A Scipy sparse adjacency matrix.
    nodes: the nodes that will be linked to the graph.
        list or np.array: the nodes connected to `nbrs_to_link`
        int: new added nodes connected to `nbrs_to_link`, 
            node ids [num_nodes, ..., num_nodes+nodes-1].            
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
    graphgallery.functional.augment_edge    

    """

    adj_matrix = adj_matrix.tocoo(copy=False)
    edge_index = adj_matrix.row, adj_matrix.col

    augmented_edge_index, augmented_edge_weight = augment_edge(edge_index, nodes,
                                                               edge_weight=adj_matrix.data,
                                                               nbrs_to_link=nbrs_to_link,
                                                               common_nbrs=common_nbrs,
                                                               fill_weight=fill_weight)

    N = augmented_edge_index.max() + 1
    augmented_adj = sp.csr_matrix((augmented_edge_weight, augmented_edge_index),
                                  shape=(N, N))

    augmented_adj.eliminate_zeros()
    return augmented_adj
