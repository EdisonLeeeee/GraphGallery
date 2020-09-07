import numpy as np
import scipy.sparse as sp
from graphgallery.transformers import Transformer
from graphgallery.utils.type_check import is_list_like
from graphgallery.utils.shape import repeat


class NormalizeAdj(Transformer):
    """Normalize adjacency matrix."""

    def __init__(self, rate=-0.5, self_loop=1.0):
        """
        # return a normalized adjacency matrix
        >>> normalize_adj(adj, rate=-0.5) 

        # return a list of normalized adjacency matrices
        >>> normalize_adj(adj, adj, rate=[-0.5, 1.0]) 

        Parameters
        ----------
            rate: Single or a list of float scale, optional.
                the normalize rate for `adj_matrics`.
            self_loop: float scalar, optional.
                weight of self loops for the adjacency matrix.
        """
        super().__init__()
        self.rate = rate
        self.self_loop = self_loop

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
            graphgallery.transformers.normalize_adj
        """
        return normalize_adj(*adj_matrics, rate=self.rate,
                             self_loop=self.self_loop)

    def __repr__(self):
        return f"{self.__class__.__name__}(normalize rate={self.rate}, self loop weight={self.self_loop })"


def normalize_adj(*adj_matrics, rate=-0.5, self_loop=1.0):
    """Normalize adjacency matrix.

    >>> normalize_adj(adj, rate=-0.5) # return a normalized adjacency matrix

    # return a list of normalized adjacency matrices
    >>> normalize_adj(adj, adj, rate=[-0.5, 1.0]) 

    Parameters
    ----------
        adj_matrics: Scipy matrix or Numpy array or a list of them 
            Single or a list of Scipy sparse matrices or Numpy arrays.
        rate: Single or a list of float scale, optional.
            the normalize rate for `adj_matrics`.
        self_loop: float scalar, optional.
            weight of self loops for the adjacency matrix.

    Returns
    ----------
        Single or a list of Scipy sparse matrix or Numpy matrices.

    See also
    ----------
        graphgallery.transformers.NormalizeAdj          

    """
    def normalize(adj, r):

        # here a new copy of adj is created
        adj = adj + self_loop * sp.eye(adj.shape[0])

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

    # TODO: check the input adj and rate
    size = len(adj_matrics)
    if size == 1:
        return normalize(adj_matrics[0], rate)
    else:
        rates = repeat(rate, size)
        return tuple(normalize(adj, r) for adj, r in zip(adj_matrics, rates))
