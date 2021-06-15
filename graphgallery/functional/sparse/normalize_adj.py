import numpy as np
import scipy.sparse as sp
import graphgallery as gg

from ..base_transforms import SparseTransform
from ..decorators import multiple
from ..transform import Transform


@Transform.register()
class NormalizeAdj(SparseTransform):
    """Normalize adjacency matrix."""

    def __init__(self, rate=-0.5, fill_weight=1.0, symmetric=True):
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
        symmetric: bool, optional
            whether to use symmetrical  normalization
    """
        super().__init__()
        self.collect(locals())

    def __call__(self, *adj_matrix):
        """
        Parameters
        ----------
        adj_matrix: Scipy matrix or Numpy array or a list of them 
            Single or a list of Scipy sparse matrices or Numpy arrays.

        Returns
        -------
        Single or a list of Scipy sparse matrix or Numpy matrices.

        See also
        ----------
        graphgallery.functional.normalize_adj
        """
        return normalize_adj(*adj_matrix,
                             rate=self.rate,
                             fill_weight=self.fill_weight,
                             symmetric=self.symmetric)


@multiple()
def normalize_adj(adj_matrix, rate=-0.5, fill_weight=1.0, symmetric=True):
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
    symmetric: bool, optional
        whether to use symmetrical  normalization

    Returns
    ----------
    Single or a list of Scipy sparse matrix or Numpy matrices.

    See also
    ----------
    graphgallery.functional.NormalizeAdj          

    """
    def _normalize_adj(adj, r):

        # here a new copy of adj is created
        if fill_weight:
            adj = adj + fill_weight * sp.eye(adj.shape[0], dtype=adj.dtype, format='csr')
        else:
            adj = adj.copy()

        if r is None:
            return adj

        degree = np.ravel(adj.sum(1))
        degree_power = np.power(degree, r)

        if sp.isspmatrix(adj):
            adj = adj.tocoo(copy=False)
            adj.data = degree_power[adj.row] * adj.data
            if symmetric:
                adj.data *= degree_power[adj.col]
            adj = adj.tocsr(copy=False)
        else:
            degree_power_matrix = sp.diags(degree_power)
            adj = degree_power_matrix @ adj
            if symmetric:
                adj = adj @ degree_power_matrix
        return adj

    if gg.is_listlike(rate):
        return tuple(_normalize_adj(adj_matrix, r) for r in rate)
    else:
        return _normalize_adj(adj_matrix, rate)


normalized_laplacian_matrix = normalize_adj
