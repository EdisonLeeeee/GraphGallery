import numpy as np
import scipy.sparse as sp

from ..transform import SparseTransform
from ..decorators import multiple
from ..transform import Transform


@Transform.register()
class AddSelfLoop(SparseTransform):
    """Add selfloops for adjacency matrix."""

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
        return add_self_loop(*adj_matrix)


@Transform.register()
class RemoveSelfLoop(SparseTransform):
    """Remove selfloops for adjacency matrix."""

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
        graphgallery.functional.eliminate_selfloops
        """
        return remove_self_loop(*adj_matrix)


@multiple()
def add_self_loop(adj_matrix: sp.csr_matrix):
    """add selfloops for adjacency matrix.

    >>>add_self_loop(adj) # return an adjacency matrix with selfloops

    # return a list of adjacency matrices with selfloops
    >>>add_self_loop(adj, adj) 

    Parameters
    ----------
    adj_matrix: Scipy matrix or Numpy array or a list of them 
        Single or a list of Scipy sparse matrices or Numpy arrays.

    Returns
    -------
    Single or a list of Scipy sparse matrix or Numpy matrices.


    """
    adj_matrix = remove_self_loop(adj_matrix)
    return adj_matrix + sp.eye(adj_matrix.shape[0], dtype=adj_matrix.dtype, format='csr')


@multiple()
def remove_self_loop(adj_matrix):
    """eliminate selfloops for adjacency matrix.

    >>>remove_self_loop(adj) # return an adjacency matrix without selfloops

    # return a list of adjacency matrices without selfloops
    >>>remove_self_loop(adj, adj) 

    Parameters
    ----------
    adj_matrix: Scipy matrix or Numpy array or a list of them 
        Single or a list of Scipy sparse matrices or Numpy arrays.

    Returns
    -------
    Single or a list of Scipy sparse matrix or Numpy matrices.


    """
    if sp.issparse(adj_matrix):
        adj_matrix = adj_matrix - sp.diags(adj_matrix.diagonal(), format='csr')
        adj_matrix.eliminate_zeros()
    else:
        adj_matrix = adj_matrix - np.diag(adj_matrix)
    return adj_matrix
