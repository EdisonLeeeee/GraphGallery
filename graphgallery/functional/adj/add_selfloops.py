import numpy as np
import scipy.sparse as sp
import graphgallery as gg

from ..transforms import Transform
from ..ops import repeat
from ..decorators import MultiInputs


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
            graphgallery.functional.add_selfloops
        """
        return add_selfloops(*adj_matrix, fill_weight=self.fill_weight)

    def __repr__(self):
        """
        Return a repr representation of a repr__.

        Args:
            self: (todo): write your description
        """
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
        graphgallery.functional.AddSelfLoops          

    """
    def _add_selfloops(adj, w):
        """
        Return a new adjops object.

        Args:
            adj: (array): write your description
            w: (todo): write your description
        """

        # here a new copy of adj is created
        if w:
            return adj + w * sp.eye(adj.shape[0], dtype=adj.dtype)
        else:
            return adj.copy()

    if gg.is_listlike(fill_weight):
        return tuple(_add_selfloops(adj_matrix, w) for w in fill_weight)
    else:
        return _add_selfloops(adj_matrix, fill_weight)
