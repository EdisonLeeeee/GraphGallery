import numpy as np
import scipy.sparse as sp

from ..base_transforms import SparseTransform
from ..decorators import multiple
from ..transform import Transform


@Transform.register()
class AdjPower(SparseTransform):
    """Computing the power of adjacency matrix."""

    def __init__(self, power: int = 1):
        """
        Parameters
        ----------
        power: int scalar, optional.
            power of the adjacency matrix.
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
        graphgallery.functional.adj_power
        """
        return adj_power(*adj_matrix, power=self.power)


@multiple()
def adj_power(adj_matrix: sp.csr_matrix, power=1):
    """Computing the power of adjacency matrix.
    
    Parameters
    ----------
    power: int scalar, optional.
        power of the adjacency matrix.
    """
    res = adj_matrix
    for _ in range(power-1):
        res = res @ adj_matrix
    return res
    
