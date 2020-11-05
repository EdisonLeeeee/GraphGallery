import numpy as np
import scipy.sparse as sp
from ..transforms import Transform
from ..ops import repeat


class SVD(Transform):

    def __init__(self, k=50, threshold=0.01, binaryzation=False):
        """
        Initialize the k - means threshold.

        Args:
            self: (todo): write your description
            k: (int): write your description
            threshold: (float): write your description
            binaryzation: (str): write your description
        """
        super().__init__()
        self.k = k
        self.threshold = threshold
        self.binaryzation = binaryzation

    def __call__(self, adj_matrix):
        """
        Call the adjacency matrix.

        Args:
            self: (todo): write your description
            adj_matrix: (array): write your description
        """
        return svd(adj_matrix, k=self.k,
                   threshold=self.threshold,
                   binaryzation=self.binaryzation)

    def __repr__(self):
        """
        Return a repr representation of a repr__.

        Args:
            self: (todo): write your description
        """
        return f"{self.__class__.__name__}(k={self.k}, threshold={self.threshold}, binaryzation={self.binaryzation})"


def svd(adj_matrix, k=50, threshold=0.01, binaryzation=False):
    """
    Compute the svd matrix.

    Args:
        adj_matrix: (array): write your description
        k: (array): write your description
        threshold: (float): write your description
        binaryzation: (todo): write your description
    """

    adj_matrix = adj_matrix.asfptype()
    U, S, V = sp.linalg.svds(adj_matrix, k=k)
    adj_matrix = (U * S) @ V

    if threshold is not None:
        # sparsification
        adj_matrix[adj_matrix <= threshold] = 0.

    adj_matrix = sp.csr_matrix(adj_matrix)

    if binaryzation:
        # TODO
        adj_matrix.data[adj_matrix.data > 0] = 1.0

    return adj_matrix
