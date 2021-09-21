import numpy as np
import scipy.sparse as sp
from utils import normalized_laplacian_matrix
from sklearn import preprocessing


class GLEE:
    r"""An implementation of `"Geometric Laplacian Eigenmaps" <https://arxiv.org/abs/1905.09763>`_
    from the Journal of Complex Networks '20 paper "GLEE: Geometric Laplacian Eigenmap Embedding".
    The procedure extracts the eigenvectors corresponding to the largest eigenvalues 
    of the graph Laplacian. These vectors are used as the node embedding.
    """

    def __init__(self, dimensions: int = 32, seed: int = None):

        self.dimensions = dimensions
        self.seed = seed

    def fit(self, graph: sp.csr_matrix):
        """
        Fitting a Geometric Laplacian EigenMaps model.
        """
        L_tilde = normalized_laplacian_matrix(graph)
        _, self._embedding = sp.linalg.eigsh(L_tilde, k=self.dimensions + 1,
                                             which='LM', return_eigenvectors=True)

    def get_embedding(self, normalize=True) -> np.array:
        """Getting the node embedding."""
        embedding = self._embedding
        if normalize:
            embedding = preprocessing.normalize(embedding)            
        return embedding

