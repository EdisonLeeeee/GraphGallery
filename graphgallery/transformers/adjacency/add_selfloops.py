import numpy as np
import scipy.sparse as sp
from graphgallery.transformers import Transformer
from graphgallery.utils.type_check import is_list_like
from graphgallery.utils.shape import repeat


class AddSelfLoops(Transformer):
    """Add self loops for adjacency matrix."""

    def __init__(self, selfloop: float = 1.0):
        """
        Parameters
        ----------
            selfloop: float scalar, optional.
                weight of self loops for the adjacency matrix.
        """
        super().__init__()
        self.selfloop = selfloop

    def __call__(self, *adj_matrics):
        """
        Parameters
        ----------
            adj_matrics: Scipy matrix or Numpy array or a list of them 
                Single or a list of Scipy sparse matrices or Numpy arrays.

        Returns
        ----------
            Single or a list of Scipy sparse matrix or Numpy matrices.

        See also
        ----------
            graphgallery.transformers.add_selfloops
        """
        return add_selfloops(*adj_matrics, selfloop=self.selfloop)

    def __repr__(self):
        return f"{self.__class__.__name__}(selfloop weight={self.selfloop})"


def add_selfloops(*adj_matrics, selfloop: float = 1.0):
    """Normalize adjacency matrix.

    >>> add_selfloops(adj, selfloop=1.0) # return a normalized adjacency matrix

    # return a list of normalized adjacency matrices
    >>> selfloop(adj, adj, selfloop=[1.0, 2.0]) 

    Parameters
    ----------
        adj_matrics: Scipy matrix or Numpy array or a list of them 
            Single or a list of Scipy sparse matrices or Numpy arrays.
        selfloop: float scalar, optional.
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
        return adj + selfloop * sp.eye(adj.shape[0])

    # TODO: check the input adj and selfloop
    size = len(adj_matrics)
    if size == 1:
        return _add_selfloops(adj_matrics[0], selfloop)
    else:
        selfloops = repeat(selfloop, size)
        return tuple(_add_selfloops(adj, w) for adj, w in zip(adj_matrics, selfloops))
