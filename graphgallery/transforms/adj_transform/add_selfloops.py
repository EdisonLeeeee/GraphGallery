import numpy as np
import scipy.sparse as sp
from graphgallery.transforms import Transform
from graphgallery.utils.type_check import is_list_like
from graphgallery.utils.shape import repeat
from graphgallery.utils.decorators import MultiInputs


class AddSelfLoops(Transform):
    """Add self loops for adjacency matrix."""

    def __init__(self, fill_weight: float = 1.0):
        """
        Parameters
        ----------
            fill_weight: float scalar, optional.
                weight of self loops for the adjacency matrix.
        """
        super().__init__()
        self.fill_weight = fill_weight

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
            graphgallery.transforms.add_selfloops
        """
        return add_selfloops(*adj_matrix, fill_weight=self.fill_weight)

    def __repr__(self):
        return f"{self.__class__.__name__}(fill_weight={self.fill_weight})"


@MultiInputs()
def add_selfloops(adj_matrix, fill_weight: float = 1.0):
    """Normalize adjacency matrix.

    >>> add_selfloops(adj, fill_weight=1.0) # return a normalized adjacency matrix

    # return a list of normalized adjacency matrices
    >>> add_selfloops(adj, adj, fill_weight=[1.0, 2.0]) 

    Parameters
    ----------
        adj_matrix: Scipy matrix or Numpy array or a list of them 
            Single or a list of Scipy sparse matrices or Numpy arrays.
        fill_weight: float scalar, optional.
            weight of self loops for the adjacency matrix.

    Returns
    ----------
        Single or a list of Scipy sparse matrix or Numpy matrices.

    See also
    ----------
        graphgallery.transforms.AddSelfLoops          

    """
    def _add_selfloops(adj, w):

        # here a new copy of adj is created
        if w:
            return adj + w * sp.eye(adj.shape[0], dtype=adj.dtype)
        else:
            return adj.copy()

    if is_list_like(fill_weight):
        return tuple(_add_selfloops(adj_matrix, w) for w in fill_weight)
    else:
        return _add_selfloops(adj_matrix, fill_weight)
