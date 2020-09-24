import numpy as np
import scipy.sparse as sp
from graphgallery.utils.decorators import MultiInputs
from graphgallery import transformers as T


class SparseReshape(T.Transformer):
    """Add self loops for adjacency matrix."""

    def __init__(self, shape: tuple = None):
        """
        Parameters
        ----------
            shape: new shape.
        """
        super().__init__()
        self.shape = shape

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
            graphgallery.transformers.sparse_reshape
        """
        return sparse_reshape(*adj_matrix, shape=self.shape)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"


@MultiInputs()
def sparse_reshape(adj_matrix, shape: tuple = None):
    """

    Parameters
    ----------
        adj_matrix: Scipy matrix or Numpy array or a list of them 
            Single or a list of Scipy sparse matrices or Numpy arrays.
        shape: new shape.

    Returns
    ----------
        Single or a list of Scipy sparse matrix or Numpy matrices.

    See also
    ----------
        graphgallery.transformers.SparseReshape          

    """
    if shape is None:
        return adj_matrix.copy()
    else:
        M1, N1 = shape
        M2, N2 = adj_matrix.shape
        assert (M1>=M2) and (N1>=N2)
        edge_index, edge_weight = T.sparse_adj_to_sparse_edges(adj_matrix)
        return sp.csr_matrix((edge_weight, edge_index), shape=shape)
        