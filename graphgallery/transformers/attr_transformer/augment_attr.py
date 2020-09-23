import numpy as np
from typing import Union
from graphgallery.utils.type_check import is_scalar_like

def augment_attr(attr_matrix: np.ndarray, N: int, 
                 weight: Union[float, list, np.ndarray] =0.):
    """Augment a specified attribute matrix.
    
    Examples
    ----------
    >>> augment_attr(attr_matrix, 10, weight=1.0)
    
    >>> augment_attr(attr_matrix, 10, weight=attr_matrix[-1])
    
    Parameters
    ----------
    attr_matrix: shape [n_nodes, n_nodes].
        A Scipy sparse adjacency matrix.
    N: number of added nodes.
        node ids [n_nodes, ..., n_nodes+N-1].   
    weight: float or 1D array.
        + float scalar: the weight for the augmented matrix
        + 1D array: repeated N times to augment the matrix.
        
        
    """
    if is_scalar_like(weight):
        M = np.zeros([N, attr_matrix.shape[1]], dtype=attr_matrix.dtype) + weight
    elif isinstance(weight, (list, np.ndarray)):
        weight = weight.astype(attr_matrix.dtype, copy=False)
        M = np.tile(weight, N).reshape([N, -1])
    else:
        raise ValueError(f"Unrecognized input: {weight}.")
        
    augmented_attr = np.vstack([attr_matrix, M])
    return augmented_attr