import numpy as np
import scipy.sparse as sp
from graphgallery.transformers import Transformer
from graphgallery.utils.type_check import is_list_like
from graphgallery.utils.shape import repeat
from graphgallery.utils.decorators import MultiInputs


class AddSelfLoops(Transformer):
    """Add self loops for adjacency matrix."""

    def __init__(self, fille_weight: float = 1.0):
        """
        Parameters
        ----------
            fille_weight: float scalar, optional.
                weight of self loops for the adjacency matrix.
        """
        super().__init__()
        self.fille_weight = fille_weight

    def __call__(self, *adj_matrix):
        """
        Parameters
        ----------
            adj_matrix: Scipy matrix or Numpy array or a list of them 
                Single or a list of Scipy sparse matrices or Numpy arrays.

        Returns
        ----------
            Single or a list of Scipy sparse matrix or Numpy matrices.

        See also
        ----------
            graphgallery.transformers.add_selfloops
        """
        return add_selfloops(*adj_matrix, fille_weight=self.fille_weight)

    def __repr__(self):
        return f"{self.__class__.__name__}(fille_weight={self.fille_weight})"


@MultiInputs()
def add_selfloops(adj_matrix, fille_weight: float = 1.0):
    """Normalize adjacency matrix.

    >>> add_selfloops(adj, fille_weight=1.0) # return a normalized adjacency matrix

    # return a list of normalized adjacency matrices
    >>> selfloop(adj, adj, fille_weight=[1.0, 2.0]) 

    Parameters
    ----------
        adj_matrix: Scipy matrix or Numpy array or a list of them 
            Single or a list of Scipy sparse matrices or Numpy arrays.
        fille_weight: float scalar, optional.
            weight of self loops for the adjacency matrix.

    Returns
    ----------
        Single or a list of Scipy sparse matrix or Numpy matrices.

    See also
    ----------
        graphgallery.transformers.NormalizeAdj          

    """
    def _add_selfloops(adj, w):

        # here a new copy of adj is created
        if w:
            return adj + w * sp.eye(adj.shape[0], dtype=adj.dtype)
        else:
            return adj.copy()

    if is_list_like(fille_weight):
        return tuple(_add_selfloops(adj_matrix, w) for w in fille_weight)
    else:
        return _add_selfloops(adj_matrix, fille_weight)
    
