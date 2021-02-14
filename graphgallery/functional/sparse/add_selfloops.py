import numpy as np
import scipy.sparse as sp
import graphgallery as gg

from ..base_transforms import SparseTransform
from ..decorators import multiple
from ..transform import Transform


@Transform.register()
class AddSelfloops(SparseTransform):
    """Add self loops for adjacency matrix."""

    def __init__(self, fill_weight: float = 1.0):
        """
        Parameters
        ----------
        fill_weight: float scalar, optional.
            weight of self loops for the adjacency matrix.
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
        graphgallery.functional.add_selfloops
        """
        return add_selfloops(*adj_matrix, fill_weight=self.fill_weight)


@multiple()
def add_selfloops(adj_matrix: sp.csr_matrix, fill_weight=1.0):
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
    -------
    Single or a list of Scipy sparse matrix or Numpy matrices.

    See also
    ----------
    graphgallery.functional.AddSelfloops          

    """
    def _add_selfloops(adj, w):
        adj = eliminate_selfloops(adj)

        if w:
            return adj + w * sp.eye(adj.shape[0], dtype=adj.dtype, format='csr')
        else:
            return adj

    if gg.is_listlike(fill_weight):
        return tuple(_add_selfloops(adj_matrix, w) for w in fill_weight)
    else:
        return _add_selfloops(adj_matrix, fill_weight)


@multiple()
def eliminate_selfloops(adj_matrix):
    if sp.issparse(adj_matrix):
        adj_matrix = adj_matrix - sp.diags(adj_matrix.diagonal(), format='csr')
        adj_matrix.eliminate_zeros()
    else:
        adj_matrix = adj_matrix - np.diag(adj_matrix)
    return adj_matrix
