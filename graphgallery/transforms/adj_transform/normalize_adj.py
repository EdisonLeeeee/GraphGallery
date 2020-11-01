import numpy as np
import scipy.sparse as sp
import graphgallery as gg
from graphgallery.transforms import Transform
from graphgallery.utils.shape import repeat
from graphgallery.utils.decorators import MultiInputs

class NormalizeAdj(Transform):
    """Normalize adjacency matrix."""

    def __init__(self, rate=-0.5, fill_weight=1.0):
        """
        # return a normalized adjacency matrix
        >>> normalize_adj(adj, rate=-0.5) 

        # return a list of normalized adjacency matrices
        >>> normalize_adj(adj, adj, rate=[-0.5, 1.0]) 

        Parameters
        ----------
            rate: Single or a list of float scale, optional.
                the normalize rate for `adj_matrix`.
            fill_weight: float scalar, optional.
                weight of self loops for the adjacency matrix.
        """
        super().__init__()
        self.rate = rate
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
            graphgallery.transforms.normalize_adj
        """
        return normalize_adj(*adj_matrix, rate=self.rate,
                             fill_weight=self.fill_weight)

    def __repr__(self):
        return f"{self.__class__.__name__}(normalize rate={self.rate}, fill_weight={self.fill_weight})"


@MultiInputs()
def normalize_adj(adj_matrix, rate=-0.5, fill_weight=1.0):
    """Normalize adjacency matrix.

    >>> normalize_adj(adj, rate=-0.5) # return a normalized adjacency matrix

    # return a list of normalized adjacency matrices
    >>> normalize_adj(adj, rate=[-0.5, 1.0]) 

    Parameters
    ----------
        adj_matrix: Scipy matrix or Numpy array or a list of them 
            Single or a list of Scipy sparse matrices or Numpy arrays.
        rate: Single or a list of float scale, optional.
            the normalize rate for `adj_matrix`.
        fill_weight: float scalar, optional.
            weight of self loops for the adjacency matrix.

    Returns
    ----------
        Single or a list of Scipy sparse matrix or Numpy matrices.

    See also
    ----------
        graphgallery.transforms.NormalizeAdj          

    """
    def _normalize_adj(adj, r):

        # here a new copy of adj is created
        if fill_weight:
            adj = adj + fill_weight * sp.eye(adj.shape[0], dtype=adj.dtype)
        else:
            adj = adj.copy()

        if r is None:
            return adj

        degree = adj.sum(1).A1
        degree_power = np.power(degree, r)
        
        if sp.isspmatrix(adj):
            adj = adj.tocoo(copy=False)
            adj.data = degree_power[adj.row] * adj.data * degree_power[adj.col]
            adj = adj.tocsr(copy=False)
        else:
            degree_power_matrix = sp.diags(degree_power)
            adj = degree_power_matrix @ adj @ degree_power_matrix
            adj = adj.A
        return adj

    if gg.is_listlike(rate):
        return tuple(_normalize_adj(adj_matrix, r) for r in rate)
    else:
        return _normalize_adj(adj_matrix, rate)
        
